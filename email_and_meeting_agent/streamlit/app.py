import os, json, base64, uuid
from typing import List, Optional, Literal
from datetime import timedelta
from email.mime.text import MIMEText

import streamlit as st
import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError
from dateutil import parser as dtparse, tz

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ---------------------------
# App / Defaults
# ---------------------------
st.set_page_config(page_title="AI Email & Meeting Agent", page_icon="📬", layout="centered")
st.title("📬 AI Email & Meeting Agent")
DEFAULT_TZ = os.getenv("DEFAULT_TZ", "Europe/Berlin")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.events",
]

# ---------------------------
# Sidebar Settings
# ---------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google API Key (Gemini)", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    model_name = st.selectbox("Gemini model", ["gemini-2.5-pro", "gemini-2.5-flash"], index=0 if DEFAULT_MODEL.endswith("pro") else 1)
    tz_name = st.text_input("Default timezone (IANA)", value=DEFAULT_TZ)
    dry_run = st.toggle("Dry-run (preview only, do not send)", value=True)
    st.caption("OAuth will prompt you in the browser the first time you run an action.")

if not api_key:
    st.info("Add your Google API Key in the sidebar to enable intent parsing.", icon="ℹ️")
else:
    genai.configure(api_key=api_key)

# ---------------------------
# OAuth helpers
# ---------------------------
@st.cache_resource
def get_credentials():
    # Requires credentials.json in working directory (download from Google Cloud Console)
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())
    return creds

@st.cache_resource(show_spinner=False)
def gmail_service(creds):
    return build("gmail", "v1", credentials=creds)

@st.cache_resource(show_spinner=False)
def calendar_service(creds):
    return build("calendar", "v3", credentials=creds)

# ---------------------------
# Schemas
# ---------------------------
class SendEmailIntent(BaseModel):
    action: Literal["send_email"]
    to: List[str]
    cc: Optional[List[str]] = Field(default_factory=list)
    subject: str
    body: str

class CreateMeetingIntent(BaseModel):
    action: Literal["create_meeting"]
    title: str
    attendees: List[str]
    # Ask LLM to return ISO 8601 local datetime like "2025-09-27T10:00"
    start: str
    duration_minutes: int = 30
    timezone: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None

Intent = SendEmailIntent | CreateMeetingIntent

# ---------------------------
# LLM parsing
# ---------------------------
def parse_intent(nl_prompt: str, model_name: str, default_tz: str) -> Intent:
    """
    Convert NL request to STRICT JSON for either send_email or create_meeting.
    Forces ISO 8601 local datetime ('YYYY-MM-DDTHH:MM') for meeting start.
    """
    system = (
        "Convert the user's request into STRICT JSON with one of two schemas.\n"
        "Output ONLY JSON (no comments, backticks, or prose).\n\n"
        "1) send_email:\n"
        "{\n"
        '  "action":"send_email",\n'
        '  "to":["addr1@example.com", ...],\n'
        '  "cc":[],\n'
        '  "subject":"...",\n'
        '  "body":"..."\n'
        "}\n\n"
        "2) create_meeting:\n"
        "{\n"
        '  "action":"create_meeting",\n'
        '  "title":"...",\n'
        '  "attendees":["addr1@example.com", ...],\n'
        '  "start":"YYYY-MM-DDTHH:MM",  // ISO local time\n'
        '  "duration_minutes":30,\n'
        f'  "timezone":"<IANA tz or omit to default to {default_tz}>",\n'
        '  "description":"...",\n'
        '  "location":"..."\n'
        "}\n"
        f'Default timezone is "{default_tz}" if not provided.\n'
        "Ensure arrays for emails even if there is a single address."
    )
    model = genai.GenerativeModel(model_name, system_instruction=system)
    resp = model.generate_content(nl_prompt)
    text = resp.text.strip()
    # Extract JSON in case of accidental code fences
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"): text.rfind("}") + 1]
    data = json.loads(text)
    # Validate
    if data.get("action") == "send_email":
        return SendEmailIntent(**data)
    elif data.get("action") == "create_meeting":
        return CreateMeetingIntent(**data)
    else:
        raise ValueError("Unknown action in parsed JSON.")

# ---------------------------
# Actions
# ---------------------------
def do_send_email(gmail, intent: SendEmailIntent):
    msg = MIMEText(intent.body, "plain", "utf-8")
    msg["To"] = ", ".join([a.strip() for a in intent.to if a.strip()])
    if intent.cc:
        msg["Cc"] = ", ".join([a.strip() for a in intent.cc if a.strip()])
    msg["Subject"] = intent.subject

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    gmail.users().messages().send(userId="me", body={"raw": raw}).execute()

def do_create_meeting(cal, intent: CreateMeetingIntent, fallback_tz: str):
    tzname = intent.timezone or fallback_tz
    start_dt = dtparse.isoparse(intent.start) if "T" in intent.start else dtparse.parse(intent.start)
    if not start_dt.tzinfo:
        start_dt = start_dt.replace(tzinfo=tz.gettz(tzname))
    end_dt = start_dt + timedelta(minutes=intent.duration_minutes)

    event = {
        "summary": intent.title,
        "description": intent.description or "",
        "location": intent.location or "",
        "start": {"dateTime": start_dt.isoformat(), "timeZone": tzname},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": tzname},
        "attendees": [{"email": a.strip()} for a in intent.attendees if a.strip()],
        "conferenceData": {
            "createRequest": {
                "requestId": str(uuid.uuid4()),
                "conferenceSolutionKey": {"type": "hangoutsMeet"}
            }
        }
    }

    created = cal.events().insert(
        calendarId="primary",
        body=event,
        conferenceDataVersion=1
    ).execute()

    meet = (
        created.get("hangoutLink")
        or (created.get("conferenceData", {}).get("entryPoints", [{}])[0].get("uri"))
    )
    return {
        "htmlLink": created.get("htmlLink"),
        "meetLink": meet,
        "start": created["start"]["dateTime"],
        "end": created["end"]["dateTime"],
    }

# ---------------------------
# UI: Prompt -> Parse -> Review -> Execute
# ---------------------------
st.subheader("1) Describe what you want to do")
nl = st.text_area(
    "Examples:\n• Email alex the weekly update and cc nina, subject 'Status', body 'All green.'\n"
    "• Schedule a 45-min sync with alex@… and nina@… tomorrow at 10:00 titled 'Latency Deep-Dive'; add a Meet link.",
    height=120,
    placeholder="Type your request…",
)

col_parse, col_clear = st.columns([1, 1])
parse_clicked = col_parse.button("Parse Intent")
if col_clear.button("Clear"):
    st.session_state.pop("intent_json", None)
    st.rerun()

if parse_clicked:
    if not api_key:
        st.error("Please provide the Google API Key first.")
    elif not nl.strip():
        st.error("Please enter a request.")
    else:
        try:
            intent_obj = parse_intent(nl.strip(), model_name, tz_name)
            st.session_state.intent_json = json.loads(intent_obj.model_dump_json())
            st.success("Parsed intent successfully.")
        except Exception as e:
            st.error(f"Couldn't parse your request: {e}")

intent_json = st.session_state.get("intent_json")
if intent_json:
    with st.expander("🔎 Show Parsed Intent (JSON)"):
        st.code(json.dumps(intent_json, indent=2), language="json")

    # Render editable form based on action
    if intent_json.get("action") == "send_email":
        st.subheader("2) Review & Edit Email")
        to_str = st.text_input("To (comma-separated)", value=", ".join(intent_json.get("to", [])))
        cc_str = st.text_input("Cc (comma-separated)", value=", ".join(intent_json.get("cc", [])))
        subject = st.text_input("Subject", value=intent_json.get("subject", ""))
        body = st.text_area("Body", value=intent_json.get("body", ""), height=160)

        st.markdown("**Preview**")
        st.write(f"**To:** {to_str}")
        if cc_str.strip():
            st.write(f"**Cc:** {cc_str}")
        st.write(f"**Subject:** {subject}")
        st.code(body)

        if dry_run:
            st.info("Dry-run is ON — nothing will be sent.", icon="🧪")

        if st.button("Send Email"):
            try:
                if dry_run:
                    st.success("✅ (Dry-run) Would send the email above.")
                else:
                    creds = get_credentials()
                    gsvc = gmail_service(creds)
                    intent = SendEmailIntent(
                        action="send_email",
                        to=[x.strip() for x in to_str.split(",") if x.strip()],
                        cc=[x.strip() for x in cc_str.split(",") if x.strip()],
                        subject=subject,
                        body=body,
                    )
                    do_send_email(gsvc, intent)
                    st.success(f"✅ Email sent to: {to_str}" + (f" (cc: {cc_str})" if cc_str else ""))
            except Exception as e:
                st.error(f"Failed to send email: {e}")

    elif intent_json.get("action") == "create_meeting":
        st.subheader("2) Review & Edit Meeting")
        title = st.text_input("Title", value=intent_json.get("title", ""))
        attendees_str = st.text_input("Attendees (comma-separated emails)", value=", ".join(intent_json.get("attendees", [])))
        start_iso = st.text_input("Start (ISO 8601, e.g., 2025-09-27T10:00)", value=intent_json.get("start", ""))
        duration = st.number_input("Duration (minutes)", min_value=5, max_value=480, value=intent_json.get("duration_minutes", 30), step=5)
        tz_chosen = st.text_input("Timezone", value=intent_json.get("timezone") or tz_name)
        description = st.text_area("Description", value=intent_json.get("description", ""), height=120)
        location = st.text_input("Location", value=intent_json.get("location", ""))

        # Preview
        try:
            sdt = dtparse.isoparse(start_iso) if "T" in start_iso else dtparse.parse(start_iso)
            if not sdt.tzinfo:
                sdt = sdt.replace(tzinfo=tz.gettz(tz_chosen))
            edt = sdt + timedelta(minutes=int(duration))
            st.markdown("**Preview**")
            st.write(f"**When:** {sdt.isoformat()} → {edt.isoformat()} ({tz_chosen})")
        except Exception:
            st.warning("Could not parse the start time. Ensure ISO format like 2025-09-27T10:00", icon="⚠️")
        st.write(f"**Attendees:** {attendees_str}")
        if location: st.write(f"**Location:** {location}")
        if description: st.code(description)

        if dry_run:
            st.info("Dry-run is ON — nothing will be created.", icon="🧪")

        if st.button("Create Meeting"):
            try:
                if dry_run:
                    st.success("✅ (Dry-run) Would create the meeting above (with a Google Meet link).")
                else:
                    creds = get_credentials()
                    cal = calendar_service(creds)
                    intent = CreateMeetingIntent(
                        action="create_meeting",
                        title=title,
                        attendees=[x.strip() for x in attendees_str.split(",") if x.strip()],
                        start=start_iso,
                        duration_minutes=int(duration),
                        timezone=tz_chosen,
                        description=description,
                        location=location,
                    )
                    result = do_create_meeting(cal, intent, tz_name)
                    st.success(f"📅 Created: {result['start']} → {result['end']}")
                    if result.get("meetLink"):
                        st.write(f"**Google Meet:** {result['meetLink']}")
                    if result.get("htmlLink"):
                        st.write(f"[Open event]({result['htmlLink']})")
            except Exception as e:
                st.error(f"Failed to create meeting: {e}")
