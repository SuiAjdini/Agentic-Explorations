from firecrawl import Firecrawl
import streamlit as st
import os
import json
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Startup Info Extraction",
    page_icon="🔍",
    layout="wide"
)

st.title("AI Startup Insight with Firecrawl & Gemini")


# ---------- Helpers for social extraction ----------

def _normalize_url(base_url: str, href: str) -> str:
    """Normalize relative / protocol-relative URLs into absolute URLs."""
    if not href:
        return ""
    href = href.strip()
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return urljoin(base_url, href)
    return href


def find_social_links(main_url: str) -> dict:
    """
    Fetch the main website HTML and auto-detect LinkedIn, X/Twitter, Crunchbase links.
    Returns a dict like:
        {"linkedin": "https://...", "twitter": "https://...", "crunchbase": "https://..."}
    """
    socials = {}
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
        resp = requests.get(main_url, headers=headers, timeout=15)
        resp.raise_for_status()
        html = resp.text

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = _normalize_url(main_url, a["href"])
            if not href.startswith("http"):
                continue

            low = href.lower()

            if "linkedin.com" in low and "linkedin" not in socials:
                socials["linkedin"] = href
            elif ("twitter.com" in low or "x.com" in low) and "twitter" not in socials:
                socials["twitter"] = href
            elif "crunchbase.com" in low and "crunchbase" not in socials:
                socials["crunchbase"] = href

        return socials
    except Exception:
        # Don't break the flow if the site blocks or something goes wrong
        return {}


# Social extraction schema for Firecrawl
social_extraction_schema = {
    "type": "object",
    "properties": {
        "page_summary": {
            "type": "string",
            "description": "High-level summary of how the company presents itself on this page"
        },
        "audience_targeted": {
            "type": "string",
            "description": "Who this page seems to be targeting (e.g., developers, enterprises, home users)"
        },
        "notable_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key claims, slogans, or value propositions mentioned on the page"
        },
        "engagement_signals": {
            "type": "string",
            "description": "Signals of traction/engagement if visible (e.g., followers, likes, activity level)"
        }
    },
    "required": ["page_summary"]
}


# ---------- Sidebar for API keys ----------

with st.sidebar:
    st.header("API Configuration")

    # Check Firecrawl Key
    env_firecrawl = os.getenv("FIRECRAWL_API_KEY")
    if env_firecrawl:
        st.success("✅ Firecrawl Key loaded from .env")
        firecrawl_api_key = env_firecrawl
    else:
        firecrawl_api_key = st.text_input("Firecrawl API Key", type="password")

    # Check Google Key
    env_google = os.getenv("GOOGLE_API_KEY")
    if env_google:
        st.success("✅ Google Key loaded from .env")
        google_api_key = env_google
    else:
        google_api_key = st.text_input("Google Gemini API Key", type="password")

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This tool extracts company information using Firecrawl "
        "and analyzes it using **Google Gemini**."
    )


# ---------- Main content ----------

st.markdown("## 🔥 Firecrawl FIRE 1 Agent Capabilities")

col1, col2 = st.columns(2)

with col1:
    st.info(
        "**Advanced Web Extraction**\n\n"
        "Firecrawl's FIRE 1 agent navigates websites to extract structured data, "
        "even from complex layouts."
    )

with col2:
    st.warning(
        "**Multi-page Processing**\n\n"
        "FIRE handles pagination to gather comprehensive data across entire websites."
    )

st.markdown("---")

st.markdown("### 🌐 Enter Website URLs")
website_urls = st.text_area(
    "Website URLs (one per line)",
    placeholder="https://example.com\nhttps://another-company.com",
)

# Main JSON schema
extraction_schema = {
    "type": "object",
    "properties": {
        "company_name": {
            "type": "string",
            "description": "The official name of the company",
        },
        "company_description": {
            "type": "string",
            "description": "What the company does and its value proposition",
        },
        "company_mission": {
            "type": "string",
            "description": "The company's mission statement",
        },
        "product_features": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key features of products/services",
        },
        "contact_phone": {
            "type": "string",
            "description": "Contact phone number if available",
        },
    },
    "required": ["company_name", "company_description", "product_features"],
}

# Custom CSS
st.markdown(
    """
<style>
.stButton button {
    background-color: #4285F4;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    border: none;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}
.stButton button:hover {
    background-color: #2b6cb0;
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Start extraction ----------

if st.button("🚀 Start Analysis", type="primary"):
    if not website_urls.strip():
        st.error("Please enter at least one website URL")
    else:
        try:
            with st.spinner("Preparing agents..."):
                # Initialize Firecrawl with the loaded key
                app = Firecrawl(api_key=firecrawl_api_key)
                urls = [url.strip() for url in website_urls.split("\n") if url.strip()]

                if not urls:
                    st.error("No valid URLs found.")
                elif not google_api_key:
                    st.warning("Please provide a Google Gemini API key.")
                else:
                    tabs = st.tabs(
                        [f"Website {i+1}: {url}" for i, url in enumerate(urls)]
                    )

                    # Initialize Gemini Agent with the loaded key
                    agno_agent = Agent(
                        model=Gemini(
                            id="gemini-2.5-flash",
                            api_key=google_api_key,
                        ),
                        instructions="""
                            You are an expert business analyst.

                            You will receive:
                            - Structured data extracted from the main website
                            - Structured extracts from social / profile pages (LinkedIn, X/Twitter, Crunchbase, etc.)

                            Your job:
                            1. Reconcile all sources (website vs socials vs profiles)
                            2. Explain what the startup really does and for whom
                            3. Highlight positioning, tone, credibility, and any interesting signals
                            4. Point out any mismatch between website claims and social signals if visible

                            Keep response under 200 words. Be specific and concrete.
                        """,
                        markdown=True,
                    )

                    for i, (url, tab) in enumerate(zip(urls, tabs)):
                        with tab:
                            st.markdown(f"### 🔍 Analyzing: {url}")
                            st.markdown(
                                "<hr style='border: 2px solid #4285F4; border-radius: 5px;'>",
                                unsafe_allow_html=True,
                            )

                            with st.spinner(f"FIRE agent extracting from {url}..."):
                                try:
                                    # Structured extraction with FIRE-1 from main website
                                    data = app.extract(
                                        urls=[url],
                                        prompt=(
                                            "Analyze this website and extract company info, "
                                            "mission, and features."
                                        ),
                                        schema=extraction_schema,
                                        agent={"model": "FIRE-1"},
                                    )

                                    if data and getattr(data, "data", None):
                                        st.subheader("📊 Extracted Information")
                                        company_data = data.data

                                        # Nicely display structured fields
                                        if "company_name" in company_data:
                                            st.markdown(
                                                f"### {company_data['company_name']}"
                                            )

                                        for key, value in company_data.items():
                                            if key == "company_name":
                                                continue
                                            display_key = (
                                                key.replace("_", " ").capitalize()
                                            )
                                            if value:
                                                if isinstance(value, list):
                                                    st.markdown(
                                                        f"**{display_key}:**"
                                                    )
                                                    for item in value:
                                                        st.markdown(f"- {item}")
                                                else:
                                                    st.markdown(
                                                        f"**{display_key}:** {value}"
                                                    )

                                        # 🌐 Auto-detect socials from the main website (HTML)
                                        st.markdown("---")
                                        st.markdown(
                                            "### 🌐 Auto-detected Social / Profile Links"
                                        )

                                        social_urls = find_social_links(url)
                                        extra_sources = []

                                        if not social_urls:
                                            st.info(
                                                "No LinkedIn / X / Crunchbase links detected on this page "
                                                "(or the site blocked / obfuscated them)."
                                            )
                                        else:
                                            for label, link in social_urls.items():
                                                st.markdown(
                                                    f"- **{label.title()}**: {link}"
                                                )

                                            # 🔥 Use Firecrawl to extract structured data from each social/profile
                                            with st.spinner(
                                                "🔥 Firecrawl analyzing detected profile sources..."
                                            ):
                                                for label, link in social_urls.items():
                                                    try:
                                                        social_data = app.extract(
                                                            urls=[link],
                                                            prompt=(
                                                                "Extract key information about the company "
                                                                "and how it presents itself on this page."
                                                            ),
                                                            schema=social_extraction_schema,
                                                            agent={"model": "FIRE-1"},
                                                        )

                                                        if social_data and getattr(
                                                            social_data, "data", None
                                                        ):
                                                            extra_sources.append(
                                                                {
                                                                    "source": label,
                                                                    "url": link,
                                                                    "data": social_data.data,
                                                                }
                                                            )
                                                        else:
                                                            extra_sources.append(
                                                                {
                                                                    "source": label,
                                                                    "url": link,
                                                                    "data": {
                                                                        "page_summary": "No structured data returned by Firecrawl."
                                                                    },
                                                                }
                                                            )
                                                    except Exception as e:
                                                        extra_sources.append(
                                                            {
                                                                "source": label,
                                                                "url": link,
                                                                "data": {
                                                                    "page_summary": f"Error extracting from {link}: {e}"
                                                                },
                                                            }
                                                        )

                                            with st.expander(
                                                "🔍 View extracted social/profile insights",
                                                expanded=False,
                                            ):
                                                for src in extra_sources:
                                                    st.markdown(
                                                        f"**{src['source'].title()}** → {src['url']}"
                                                    )
                                                    st.json(src["data"])

                                        # Build combined context for Gemini
                                        combined_context = {
                                            "primary_website": url,
                                            "structured_data": company_data,
                                            "additional_social_data": extra_sources,
                                        }

                                        # 🧠 Gemini analysis
                                        try:
                                            with st.spinner(
                                                "Generating multi-source Gemini analysis..."
                                            ):
                                                agent_response = agno_agent.run(
                                                    "You will receive structured data "
                                                    "from the main website plus structured extracts "
                                                    "from social/profile pages. "
                                                    "Use all of them to understand what this startup "
                                                    "actually does, who they target, how they position "
                                                    "themselves, and how credible they look."
                                                    f"\n\nDATA:\n{json.dumps(combined_context)}"
                                                )

                                            st.subheader(
                                                "🧠 Gemini Multi-Source Analysis"
                                            )
                                            st.markdown(agent_response.content)
                                        except Exception as e:
                                            st.error(
                                                f"Gemini analysis failed: {e}"
                                            )

                                        with st.expander("🔍 View Raw Firecrawl Data"):
                                            st.json(
                                                data.data if hasattr(data, "data") else data
                                            )
                                    else:
                                        st.error(f"No data extracted from {url}.")

                                except Exception as e:
                                    st.error(f"Error processing {url}: {str(e)}")
        except Exception as e:
            st.error(f"Error during extraction: {str(e)}")
