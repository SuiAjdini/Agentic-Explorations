import os
import re
import requests
import streamlit as st
from typing import Tuple, Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env into os.environ
api_key = os.getenv("GOOGLE_API_KEY", "")


# ----------------------------
# Gemini helpers (sync)
# ----------------------------
def _invoke_text(llm: ChatGoogleGenerativeAI, prompt: str) -> str:
    msg = llm.invoke(prompt)
    # langchain_google_genai returns an AIMessage; .content is usually a string
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        # Rare: multimodal chunks; join any text parts
        content = " ".join(str(x) for x in content)
    return (content or "").strip()

def pick_action_verb(llm: ChatGoogleGenerativeAI, query: str) -> str:
    prompt = (
        "From this text, extract ONE plain English action verb that best represents the main action. "
        "Return only the verb, lowercase, no punctuation.\n"
        f"Text: {query}"
    )
    out = _invoke_text(llm, prompt)
    verb = (out.split()[0].lower() if out else "laugh")
    return re.sub(r"[^a-z]", "", verb) or "laugh"

def choose_template(llm: ChatGoogleGenerativeAI, query: str, memes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick a single template from top N memes using Gemini."""
    top = memes[:125]
    options = "\n".join([f"{m['id']} :: {m['name']}" for m in top])
    prompt = (
        "Pick EXACTLY ONE meme template ID from the list that best fits the topic.\n"
        "Return only the ID, nothing else.\n\n"
        f"Topic: {query}\n\n"
        f"Options:\n{options}\n\n"
        "Answer with only the ID:"
    )
    out = _invoke_text(llm, prompt)
    chosen = out.strip().split()[0]
    picked = next((m for m in top if m["id"] == chosen or m["name"].lower() == out.lower()), None)
    return picked or top[0]

def write_captions(llm: ChatGoogleGenerativeAI, query: str, template_name: str) -> Tuple[str, str]:
    prompt = (
        "Write concise meme captions (setup + punchline) for the topic and template.\n"
        "Rules:\n"
        "- Top: 6–10 words, Bottom: 5–8 words\n"
        "- No quotes, no emojis, plain ASCII\n"
        "- Be witty but clear.\n\n"
        f"Topic: {query}\n"
        f"Template: {template_name}\n\n"
        "Return in this format exactly:\n"
        "TOP: <top line>\n"
        "BOTTOM: <bottom line>"
    )
    out = _invoke_text(llm, prompt)
    top = re.search(r"(?im)^top:\s*(.+)$", out)
    bottom = re.search(r"(?im)^bottom:\s*(.+)$", out)
    return (
        top.group(1).strip() if top else "when it looks easy",
        bottom.group(1).strip() if bottom else "but explodes in prod",
    )


# ----------------------------
# Imgflip API
# ----------------------------
def fetch_memes() -> List[Dict[str, Any]]:
    r = requests.get("https://api.imgflip.com/get_memes", timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"Imgflip get_memes failed: {data}")
    return data["data"]["memes"]

def caption_image(template_id: str, username: str, password: str, text0: str, text1: str) -> str:
    payload = {
        "template_id": template_id,
        "username": username,
        "password": password,
        "text0": text0,
        "text1": text1,
    }
    r = requests.post("https://api.imgflip.com/caption_image", data=payload, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"Imgflip caption_image failed: {data}")
    return data["data"]["url"]  # https://i.imgflip.com/xxxx.jpg


# ----------------------------
# App
# ----------------------------
def main():
    st.title("🥸 Gemini Meme Generator")
    st.caption("Gemini picks verb/template/captions. Imgflip API renders the meme")

    with st.sidebar:
        st.subheader("⚙️ Gemini")
        model_name = st.selectbox(
            "Model",
            ["gemini-2.5-flash"],
            index=0
        )
        st.subheader("🔑 Imgflip")
        st.caption("Required by Imgflip to generate the image.")
        imgflip_user = st.text_input("Imgflip Username")
        imgflip_pass = st.text_input("Imgflip Password", type="password")

    query = st.text_input(
        "Describe your meme idea",
        placeholder="“PM says ‘small change’, server team sees a 2-hour outage”"
    )

    if st.button("Generate Meme 🚀"):
        if not imgflip_user or not imgflip_pass:
            st.warning("Please provide Imgflip username & password (needed to generate).")
            st.stop()
        if not query.strip():
            st.warning("Please enter a meme idea.")
            st.stop()

        # Ensure SDK sees the key (some env-based setups rely on this)
        os.environ["GOOGLE_API_KEY"] = api_key

        try:
            with st.spinner("Thinking with Gemini…"):
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    api_key=api_key,
                    temperature=0.2,
                )
                memes = fetch_memes()
                verb = pick_action_verb(llm, query)
                picked = choose_template(llm, f"{query} (verb: {verb})", memes)
                top, bottom = write_captions(llm, query, picked["name"])

            with st.expander("Chosen Template & Captions", expanded=False):
                st.write(f"Template: **{picked['name']}** (`{picked['id']}`)")
                st.write(f"Top: {top}")
                st.write(f"Bottom: {bottom}")

            with st.spinner("Calling Imgflip API…"):
                url = caption_image(picked["id"], imgflip_user, imgflip_pass, top, bottom)

            st.success("✅ Meme generated!")
            st.image(url, caption="Generated Meme", use_container_width=True)
            st.markdown(f"**Direct Link:** [{url}]({url})  \n**Embed URL:** `{url}`")

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
