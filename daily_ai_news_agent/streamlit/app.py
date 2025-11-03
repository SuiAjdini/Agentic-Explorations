"""
Daily AI News Digest — Streamlit App
Stack: Streamlit + Tavily Search API + Google Gemini (google-generativeai)
"""

import os, json, textwrap, requests, streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from dotenv import load_dotenv, find_dotenv
ENV_PATH = os.getenv("DOTENV_PATH") or find_dotenv(usecwd=True)
if ENV_PATH:
    load_dotenv(ENV_PATH)

TAVILY_ENDPOINT = "https://api.tavily.com/search"

class TavilyResult:
    def __init__(self, title: str = "", url: str = "", content: str = "", score: Optional[float] = None, published_date: Optional[str] = None):
        self.title = title
        self.url = url
        self.content = content
        self.score = score
        self.published_date = published_date

def tavily_search(query: str, key: str, max_results: int = 8, days: int = 2):
    payload = {"api_key": key, "query": query, "search_depth": "advanced", "max_results": max_results, "days": days}
    resp = requests.post(TAVILY_ENDPOINT, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=40)
    resp.raise_for_status()
    return resp.json()

def init_gemini(api_key: str):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai

def summarize_with_gemini(genai, model_name: str, prompts: List[Tuple[str, List[TavilyResult]]], style: str, language: str, max_words: int):
    sections, footnotes, idx = [], [], 1
    for topic, results in prompts:
        results = results[:5]
        sections.append(f"## {topic}\n")
        for r in results:
            snippet = textwrap.shorten(r.content or "", width=240, placeholder="…")
            sections.append(f"- {r.title or '(no title)'} [{idx}] — {snippet}")
            footnotes.append(f"[{idx}] {r.title} — {r.url}")
            idx += 1
        sections.append("")

    system_msg = f"You are an expert news editor. Create a concise morning briefing in {language}. Keep under {max_words} words total. Style: {style}."
    user_msg = f"Sources grouped by topic:\n\n{os.linesep.join(sections)}\n\nFootnotes:\n{os.linesep.join(footnotes)}"

    model = genai.GenerativeModel(model_name, system_instruction=system_msg)
    resp = model.generate_content(user_msg)
    return getattr(resp, "text", "Error generating summary.")

def render_sources(sources: Dict[str, List[TavilyResult]]):
    with st.expander("Show fetched sources"):
        for topic, items in sources.items():
            st.markdown(f"### {topic}")
            for i, r in enumerate(items, 1):
                st.markdown(f"{i}. [{r.title}]({r.url}) — {r.published_date or ''}")

st.set_page_config(page_title="Daily AI News Digest", page_icon="🗞️", layout="wide")
st.title("🗞️ Daily AI News Digest — Gemini + Tavily")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not TAVILY_API_KEY or not GEMINI_API_KEY:
    st.error("Missing API keys — ensure your .env defines TAVILY_API_KEY and GEMINI_API_KEY or GOOGLE_API_KEY.")
    st.stop()

with st.sidebar:
    st.caption(":grey[Keys status]")
    st.write(f"Tavily: {'✅' if bool(os.getenv('TAVILY_API_KEY') or st.secrets.get('TAVILY_API_KEY')) else '❌'}  |  Gemini/Google: {'✅' if bool(os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') or st.secrets.get('GOOGLE_API_KEY')) else '❌'}")
    if ENV_PATH:
        st.caption(f":grey[.env: {ENV_PATH}]")

    topics = st.text_area("Topics (one per line)", value="AI & LLMs\nCloud & DevOps\nCybersecurity\nStartups & VC\nData Engineering", height=130).strip().splitlines()
    days = st.slider("Look back (days)", 1, 14, 2)
    max_results = st.slider("Max results per topic", 3, 15, 8)
    language = st.selectbox("Language", ["English", "Deutsch", "Français", "Italiano"], 0)
    style = st.selectbox("Tone", ["neutral, matter-of-fact", "executive, crisp", "analytical, data-first", "casual, friendly"], 1)
    max_words = st.slider("Max words (overall)", 150, 1200, 450, 50)
    model_name = st.text_input("Gemini model", value="gemini-1.5-pro")
    run_btn = st.button("Generate Briefing", type="primary")

if run_btn:
    st.info("Fetching news from Tavily…")
    gathered: Dict[str, List[TavilyResult]] = {}
    try:
        for topic in topics:
            data = tavily_search(f"latest news {topic}", TAVILY_API_KEY, max_results, days)
            results = data.get("results", [])
            gathered[topic] = [TavilyResult(it.get("title"), it.get("url"), it.get("content"), it.get("score"), it.get("published_date")) for it in results]
    except Exception as e:
        st.error(f"Tavily error: {type(e).__name__}: {e}")
        st.stop()

    render_sources(gathered)

    st.info("Summarizing with Gemini…")
    try:
        genai = init_gemini(GEMINI_API_KEY)
        summary = summarize_with_gemini(genai, model_name, list(gathered.items()), style, language, max_words)
    except Exception as e:
        st.error(f"Gemini error: {type(e).__name__}: {e}")
        st.stop()

    st.markdown(summary)
    st.download_button("Download Summary", data=summary.encode(), file_name="daily_briefing.md", mime="text/markdown")