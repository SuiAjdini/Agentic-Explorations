import os, json, base64, uuid, re, pathlib, datetime
from typing import List, Optional, Literal, Dict
from datetime import timedelta, datetime as dt
from email.mime.text import MIMEText

import streamlit as st
import google.generativeai as genai
from pydantic import BaseModel, Field
from dateutil import parser as dtparse, tz

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# =========================
# App / Defaults
# =========================
st.set_page_config(page_title="AI Email & Meeting Agent", page_icon="📬", layout="wide")
st.title("📬 AI Email & Meeting Agent")

DEFAULT_TZ = os.getenv("DEFAULT_TZ", "Europe/Berlin")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# --- Scopes ---
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
]

SUPPORTED_LANGS = {
    "auto": "Auto",
    "en": "English",
    "de": "Deutsch",
    "es": "Español",
}

TEMPLATES_DIR = pathlib.Path("templates")
EMAIL_TEMPLATES_DIR = TEMPLATES_DIR / "email"
MEETING_TEMPLATES_DIR = TEMPLATES_DIR / "meeting"
EMAIL_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
MEETING_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


def ensure_sample_email_templates():
    samples = {
        "weekly_status": {
            "subject": "Weekly Status – {project} – Week {week_no}",
            "body": (
                "Hi {first_name},\n\n"
                "Quick update on {project}:\n"
                "- Highlights: {highlights}\n"
                "- Risks/Blockers: {risks}\n"
                "- Next steps (by {next_step_date}): {next_steps}\n\n"
                "Metrics:\n"
                "- Velocity: {velocity}\n"
                "- Bugs closed: {bugs_closed}\n"
                "- Uptime: {uptime}\n\n"
                "Best,\n{sender_name}"
            )
        },
        "intro_outreach": {
            "subject": "Intro: {your_company} × {their_company}",
            "body": (
                "Hello {first_name},\n\n"
                "I’m {sender_name} from {your_company}. We’re working on {what_you_do} and I thought it might be relevant for {their_company}.\n\n"
                "Here’s why it could help:\n"
                "- {benefit_1}\n"
                "- {benefit_2}\n"
                "- {benefit_3}\n\n"
                "Would you be open to a {duration}-minute chat next week?\n\n"
                "Best regards,\n{sender_name}\n{sender_role}\n{sender_contact}"
            )
        }
    }
    for name, data in samples.items():
        p = EMAIL_TEMPLATES_DIR / f"{name}.json"
        if not p.exists():
            p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

ensure_sample_email_templates()

# =========================
# Sidebar Settings
# =========================
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google API Key (Gemini)", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    model_name = st.selectbox("Gemini model", ["gemini-2.5-flash", "gemini-2.5-flash"], index=0 if DEFAULT_MODEL.endswith("pro") else 1)
    tz_name = st.text_input("Default timezone (IANA)", value=DEFAULT_TZ)
    dry_run = st.toggle("Dry-run (preview only, do not send)", value=True)
    st.caption("OAuth will prompt you in the browser the first time you run an action.")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.info("Add your Google API Key in the sidebar to enable LLM features (parsing, translate, rewrite).", icon="ℹ️")

# =========================
# OAuth helpers
# =========================
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

# =========================
# Schemas
# =========================
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

# =========================
# LLM helpers (translate / tone / detect)
# =========================
def _model(system_instruction: Optional[str] = None):
    if not api_key:
        raise RuntimeError("Gemini API key is required for this feature.")
    if system_instruction:
        return genai.GenerativeModel(model_name, system_instruction=system_instruction)
    return genai.GenerativeModel(model_name)

def gemini_detect_language(text: str) -> str:
    """Return BCP-47-ish short code guess (en/de/es)."""
    sys = (
        "You are a language detector. Reply with ONLY a language code among: en, de, es.\n"
        "If unclear, choose the closest. Output exactly one of: en, de, es."
    )
    resp = _model(sys).generate_content(text[:4000])
    code = (resp.text or "en").strip().lower()
    return "en" if code not in {"en","de","es"} else code

def gemini_translate(text: str, target_lang: str) -> str:
    if target_lang == "auto":
        return text
    sys = f"Translate the following text into {SUPPORTED_LANGS[target_lang]}. Keep meaning, tone, and formatting. Output only the translated text."
    resp = _model(sys).generate_content(text[:10000])
    return (resp.text or text).strip()

def gemini_rewrite(text: str, style: Literal["shorten","polish","empathetic"]) -> str:
    if style == "shorten":
        prompt = "Rewrite the email to be significantly shorter while preserving key points. Keep it professional."
    elif style == "polish":
        prompt = "Rewrite the email to be clear, concise, and professional. Improve grammar and flow without changing meaning."
    else:
        prompt = "Rewrite the email with an empathetic, supportive tone while staying professional and concise."
    sys = "You are a helpful writing assistant. Output only the rewritten email body."
    resp = _model(sys).generate_content(f"{prompt}\n\n---\n{text[:10000]}")
    return (resp.text or text).strip()

# =========================
# Intent parsing (unchanged core)
# =========================
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
        '  "start":"YYYY-MM-DDTHH:MM",\n'
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
    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"): text.rfind("}") + 1]
    data = json.loads(text)
    if data.get("action") == "send_email":
        return SendEmailIntent(**data)
    elif data.get("action") == "create_meeting":
        return CreateMeetingIntent(**data)
    else:
        raise ValueError("Unknown action in parsed JSON.")

# =========================
# Actions
# =========================
def do_send_email(gmail, intent: SendEmailIntent, subject_override: Optional[str] = None, body_override: Optional[str] = None, to_override: Optional[List[str]] = None):
    subject = subject_override if subject_override is not None else intent.subject
    body = body_override if body_override is not None else intent.body
    to_list = to_override if to_override is not None else intent.to

    msg = MIMEText(body, "plain", "utf-8")
    msg["To"] = ", ".join([a.strip() for a in to_list if a.strip()])
    if intent.cc:
        msg["Cc"] = ", ".join([a.strip() for a in intent.cc if a.strip()])
    msg["Subject"] = subject

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

# =========================
# Templates (save/load/apply)
# =========================
def list_templates(kind: Literal["email","meeting"]) -> List[str]:
    d = EMAIL_TEMPLATES_DIR if kind == "email" else MEETING_TEMPLATES_DIR
    return [p.stem for p in sorted(d.glob("*.json"))]

def load_template(kind: Literal["email","meeting"], name: str) -> Dict:
    d = EMAIL_TEMPLATES_DIR if kind == "email" else MEETING_TEMPLATES_DIR
    p = d / f"{name}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def save_template(kind: Literal["email","meeting"], name: str, data: Dict):
    d = EMAIL_TEMPLATES_DIR if kind == "email" else MEETING_TEMPLATES_DIR
    p = d / f"{name}.json"
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def substitute_placeholders(text: str, variables: Dict[str, str]) -> str:
    def repl(match):
        key = match.group(1)
        return variables.get(key, match.group(0))
    return re.sub(r"\{([a-zA-Z0-9_]+)\}", repl, text)

# =========================
# Daily Digest
# =========================
def fetch_unread_emails(gsvc, max_count=20, newer_than_days=1):
    q = f"is:unread newer_than:{newer_than_days}d"
    res = gsvc.users().messages().list(userId="me", q=q, maxResults=max_count).execute()
    messages = res.get("messages", []) or []
    out = []
    for m in messages:
        full = gsvc.users().messages().get(userId="me", id=m["id"], format="full").execute()
        snippet = full.get("snippet", "")
        headers = full.get("payload", {}).get("headers", [])
        subj = next((h["value"] for h in headers if h["name"].lower()=="subject"), "(no subject)")
        frm = next((h["value"] for h in headers if h["name"].lower()=="from"), "")
        date = next((h["value"] for h in headers if h["name"].lower()=="date"), "")
        out.append({"from": frm, "subject": subj, "snippet": snippet, "date": date})
    return out

def fetch_todays_events(cal, tz_name):
    now = dt.now(tz.gettz(tz_name))
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    events = cal.events().list(
        calendarId="primary",
        timeMin=start.isoformat(),
        timeMax=end.isoformat(),
        singleEvents=True,
        orderBy="startTime"
    ).execute().get("items", [])
    parsed = []
    for e in events:
        start_dt = e.get("start", {}).get("dateTime") or e.get("start", {}).get("date")
        end_dt = e.get("end", {}).get("dateTime") or e.get("end", {}).get("date")
        parsed.append({
            "summary": e.get("summary","(no title)"),
            "start": start_dt,
            "end": end_dt,
            "link": e.get("htmlLink","")
        })
    return parsed

def make_digest_draft(unread: List[Dict], events: List[Dict]) -> Dict[str,str]:
    # Ask LLM to summarize emails + extract action items
    bullets_emails = "\n".join([f"- {u['subject']} — {u['from']}: {u['snippet']}" for u in unread]) or "- (No unread emails)"
    bullets_events = "\n".join([f"- {e['start']} → {e['end']}: {e['summary']}" for e in events]) or "- (No events)"
    prompt = (
        "Summarize these unread emails and extract clear action items (owner, task, due if present). "
        "Return two sections: 'Summary' and 'Action Items' (bulleted).\n\n"
        f"Unread emails:\n{bullets_emails}\n\n"
        "Meetings today:\n" + bullets_events
    )
    resp = _model().generate_content(prompt)
    body = resp.text or "Summary:\n- No data\n\nAction Items:\n- None"
    subject = f"Daily Digest – {dt.now().date().isoformat()}"
    return {"subject": subject, "body": body}

# =========================
# UI: Prompt -> Parse -> Review -> Execute (+ new features)
# =========================
st.subheader("1) Describe what you want to do")
nl = st.text_area(
    "Examples:\n• Email alex the weekly update and cc nina, subject 'Status', body 'All green.'\n"
    "• Schedule a 45-min sync with alex@… and nina@… tomorrow at 10:00 titled 'Latency Deep-Dive'; add a Meet link.",
    height=120,
    placeholder="Type your request…",
)

col_parse, col_clear, col_digest = st.columns([1,1,1])
parse_clicked = col_parse.button("Parse Intent")
if col_clear.button("Clear"):
    st.session_state.pop("intent_json", None)
    st.rerun()

# --- Daily Digest ---
if col_digest.button("Daily Digest (Unread + Today's Meetings)"):
    try:
        creds = get_credentials()
        gsvc = gmail_service(creds)
        cal = calendar_service(creds)

        with st.spinner("Collecting unread emails and today's meetings…"):
            unread = fetch_unread_emails(gsvc, max_count=25, newer_than_days=2)
            events = fetch_todays_events(cal, tz_name)
            digest = make_digest_draft(unread, events)

        st.success("Digest drafted. You can send it below.")
        st.session_state.intent_json = SendEmailIntent(
            action="send_email",
            to=["me"],  # You can change below
            cc=[],
            subject=digest["subject"],
            body=digest["body"],
        ).model_dump()
    except Exception as e:
        st.error(f"Failed to create daily digest: {e}")

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


    # ============= EMAIL FLOW =============
    if intent_json.get("action") == "send_email":
        st.subheader("2) Review & Edit Email")

        # ----- Templates (Email) -----
        with st.expander("📄 Templates (Email)"):
            tcol1, tcol2, tcol3 = st.columns([2,1,1])
            existing = ["(none)"] + list_templates("email")
            chosen_template = tcol1.selectbox("Load template", existing, index=0)
            template_name_new = tcol2.text_input("Save as (name)")

            # Ensure session state defaults BEFORE widgets that bind to these keys
            if "email_subject" not in st.session_state:
                st.session_state["email_subject"] = intent_json.get("subject", "")
            if "email_body" not in st.session_state:
                st.session_state["email_body"] = intent_json.get("body", "")

            if tcol3.button("Save Template"):
                data = {
                    "subject": st.session_state["email_subject"],
                    "body": st.session_state["email_body"],
                }
                if not template_name_new.strip():
                    st.warning("Please provide a template name.")
                else:
                    save_template("email", template_name_new.strip(), data)
                    st.success(f"Saved template '{template_name_new.strip()}'")

            # Apply template (update state FIRST, then widgets render)
            if chosen_template != "(none)":
                tdata = load_template("email", chosen_template)
                if tdata:
                    st.markdown("**Template variables** (format: key=value per line)")
                    vars_text = st.text_area(
                        "Variables",
                        value=st.session_state.get("email_vars_text", "first_name=Alex\nproject=Migration"),
                        height=90,
                        key="email_vars_text_area",
                    )
                    variables = {}
                    for line in vars_text.splitlines():
                        if "=" in line:
                            k, v = line.split("=", 1)
                            variables[k.strip()] = v.strip()
                    tpl_subject = tdata.get("subject", "")
                    tpl_body = tdata.get("body", "")
                    st.session_state["email_subject"] = substitute_placeholders(tpl_subject, variables)
                    st.session_state["email_body"] = substitute_placeholders(tpl_body, variables)
                    st.info("Applied template (you can still edit in the fields below).")

        # Editable fields
        to_str = st.text_input("To (comma-separated)", value=", ".join(intent_json.get("to", [])), key="email_to")
        cc_str = st.text_input("Cc (comma-separated)", value=", ".join(intent_json.get("cc", [])), key="email_cc")
        subject = st.text_input("Subject", value=st.session_state["email_subject"], key="email_subject")
        body = st.text_area("Body", value=st.session_state["email_body"], height=220, key="email_body")

        # ----- One-click Improve (use callbacks) -----
        def _rewrite_body(style):
            try:
                st.session_state["email_body"] = gemini_rewrite(st.session_state["email_body"], style)
            except Exception as e:
                st.session_state["rewrite_error"] = str(e)

        icol1, icol2, icol3 = st.columns([1,1,1])
        icol1.button("✨ Shorten", on_click=_rewrite_body, args=("shorten",))
        icol2.button("🧼 Polish", on_click=_rewrite_body, args=("polish",))
        icol3.button("💜 More empathetic", on_click=_rewrite_body, args=("empathetic",))

        if "rewrite_error" in st.session_state:
            st.error(f"Rewrite failed: {st.session_state['rewrite_error']}")
            st.session_state.pop("rewrite_error", None)

        st.markdown("**Preview**")
        st.write(f"**To:** {to_str}")
        if cc_str.strip():
            st.write(f"**Cc:** {cc_str}")
        st.write(f"**Subject:** {st.session_state['email_subject']}")
        st.code(st.session_state["email_body"])

        # ----- Auto-translate & Per-recipient Language -----
        st.markdown("---")
        st.markdown("### 🌐 Auto-translate & Per-recipient Language")
        try:
            detected_lang = gemini_detect_language(st.session_state["email_body"]) if api_key else "en"
        except Exception:
            detected_lang = "en"
        st.write(f"Detected language for current body: **{SUPPORTED_LANGS.get(detected_lang, 'English')}**")

        recipient_list = [x.strip() for x in to_str.split(",") if x.strip()]
        lang_map = {}
        lang_cols = st.columns(min(3, max(1, len(recipient_list))) or 1)
        for idx, r in enumerate(recipient_list):
            sel = lang_cols[idx % len(lang_cols)].selectbox(
                f"{r} language",
                options=list(SUPPORTED_LANGS.keys()),
                index=list(SUPPORTED_LANGS.keys()).index(detected_lang if detected_lang in SUPPORTED_LANGS else "en"),
                format_func=lambda k: SUPPORTED_LANGS[k],
                key=f"lang_{r}"
            )
            lang_map[r] = sel

        # Build unique language sets and translated versions
        unique_langs = sorted(set(lang_map.values()))
        if "translations" not in st.session_state:
            st.session_state["translations"] = {}

        if st.button("🔁 Generate Translated Versions"):
            try:
                translations = {}
                for lang in unique_langs:
                    if lang == "auto":
                        translations[lang] = {"subject": st.session_state["email_subject"], "body": st.session_state["email_body"]}
                    else:
                        sub_trans = gemini_translate(st.session_state["email_subject"], lang)
                        body_trans = gemini_translate(st.session_state["email_body"], lang)
                        translations[lang] = {"subject": sub_trans, "body": body_trans}
                st.session_state["translations"] = translations
                st.success("Generated translations.")
            except Exception as e:
                st.error(f"Translation failed: {e}")

        translations = st.session_state.get("translations", {})
        if translations:
            st.markdown("**Translated Previews**")
            for lang, data in translations.items():
                st.write(f"— **{SUPPORTED_LANGS[lang]}**")
                st.write(f"**Subject:** {data['subject']}")
                st.code(data["body"])

        if dry_run:
            st.info("Dry-run is ON — nothing will be sent.", icon="🧪")

        if st.button("Send Email(s)"):
            try:
                if dry_run:
                    st.success("✅ (Dry-run) Would send localized emails per recipient language mapping.")
                else:
                    creds = get_credentials()
                    gsvc = gmail_service(creds)

                    # Group recipients by chosen language
                    buckets: Dict[str, List[str]] = {}
                    for r, lang in lang_map.items():
                        buckets.setdefault(lang, []).append(r)

                    # Default if no translations created yet
                    translations = st.session_state.get("translations", {})
                    if not translations:
                        translations = {}
                        for lang in buckets.keys():
                            if lang == "auto":
                                translations[lang] = {
                                    "subject": st.session_state["email_subject"],
                                    "body": st.session_state["email_body"]
                                }
                            else:
                                translations[lang] = {
                                    "subject": gemini_translate(st.session_state["email_subject"], lang),
                                    "body": gemini_translate(st.session_state["email_body"], lang)
                                }

                    # Send one email per language bucket
                    for lang, recipients in buckets.items():
                        data = translations.get(lang, {
                            "subject": st.session_state["email_subject"],
                            "body": st.session_state["email_body"]
                        })
                        intent = SendEmailIntent(
                            action="send_email",
                            to=recipients,
                            cc=[x.strip() for x in cc_str.split(",") if x.strip()],
                            subject=data["subject"],
                            body=data["body"],
                        )
                        do_send_email(gsvc, intent)
                    st.success(f"✅ Sent {len(buckets)} localized email(s).")
            except Exception as e:
                st.error(f"Failed to send email(s): {e}")

    # ============= MEETING FLOW =============
    elif intent_json.get("action") == "create_meeting":
        st.subheader("2) Review & Edit Meeting")

        # ----- Templates (Meeting) -----
        with st.expander("📄 Templates (Meeting)"):
            mcol1, mcol2, mcol3 = st.columns([2,1,1])
            existing = ["(none)"] + list_templates("meeting")
            chosen_template = mcol1.selectbox("Load template", existing, index=0)
            template_name_new = mcol2.text_input("Save as (name)", key="meet_tpl_name")
            if mcol3.button("Save Template"):
                data = {
                    "title": st.session_state.get("meet_title", intent_json.get("title", "")),
                    "description": st.session_state.get("meet_description", intent_json.get("description", "")),
                    "location": st.session_state.get("meet_location", intent_json.get("location", "")),
                }
                if not template_name_new.strip():
                    st.warning("Please provide a template name.")
                else:
                    save_template("meeting", template_name_new.strip(), data)
                    st.success(f"Saved template '{template_name_new.strip()}'")

            if chosen_template != "(none)":
                tdata = load_template("meeting", chosen_template)
                if tdata:
                    st.markdown("**Template variables** (format: key=value per line)")
                    vars_text = st.text_area(
                        "Variables",
                        value=st.session_state.get("meet_vars_text", "project=Latency\nfirst_name=Alex"),
                        height=90,
                        key="meet_vars_text_area",
                    )
                    variables = {}
                    for line in vars_text.splitlines():
                        if "=" in line:
                            k,v = line.split("=",1)
                            variables[k.strip()] = v.strip()
                    intent_json["title"] = substitute_placeholders(tdata.get("title",""), variables) or intent_json.get("title","")
                    intent_json["description"] = substitute_placeholders(tdata.get("description",""), variables) or intent_json.get("description","")
                    intent_json["location"] = substitute_placeholders(tdata.get("location",""), variables) or intent_json.get("location","")

        title = st.text_input("Title", value=intent_json.get("title", ""), key="meet_title")
        attendees_str = st.text_input("Attendees (comma-separated emails)", value=", ".join(intent_json.get("attendees", [])))
        start_iso = st.text_input("Start (ISO 8601, e.g., 2025-09-27T10:00)", value=intent_json.get("start", ""))
        duration = st.number_input("Duration (minutes)", min_value=5, max_value=480, value=intent_json.get("duration_minutes", 30), step=5)
        tz_chosen = st.text_input("Timezone", value=intent_json.get("timezone") or tz_name)
        description = st.text_area("Description", value=intent_json.get("description", ""), height=120, key="meet_description")
        location = st.text_input("Location", value=intent_json.get("location", ""), key="meet_location")

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
