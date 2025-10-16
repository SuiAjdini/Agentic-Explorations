import os
import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Core LangChain components
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, tool
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader

# Load environment variables
load_dotenv()

# =========================
# Utility & Data Structures
# =========================

@dataclass
class Product:
    name: str
    price: Optional[float]
    currency: Optional[str]
    review_score: Optional[float]
    availability: Optional[str]
    shop: Optional[str]
    url: Optional[str]
    specs: Optional[Dict[str, Any]] = None
    image: Optional[str] = None
    og_title: Optional[str] = None

def _to_float_price(s: Optional[str]) -> Optional[float]:
    """Convert strings like '199,99 €' or '€199.99' to 199.99"""
    if not s:
        return None
    try:
        t = s
        # normalize whitespace and currency
        t = t.replace("EUR", "").replace("€", "€ ").replace("\u00a0", " ").strip()
        # find first number with optional , or . decimals
        m = re.search(r"(-?\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d{2})|-?\d+(?:[.,]\d{2}))", t)
        if not m:
            return None
        n = m.group(1)
        n = n.replace(" ", "").replace(".", "").replace(",", ".")
        return float(n)
    except Exception:
        return None

def _domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")
    except Exception:
        return ""

# ---------- OG metadata (cached) ----------

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_og_metadata(url: str, timeout: int = 5) -> Dict[str, Optional[str]]:
    """Fetch basic Open Graph metadata (title, image, description)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SmartShopperBot/1.0)"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")  # faster parser
        def pick(*props):
            for p in props:
                tag = soup.find("meta", property=p) or soup.find("meta", attrs={"name": p})
                if tag and tag.get("content"):
                    return tag["content"]
            return None
        og_title = pick("og:title", "twitter:title") or (soup.title.string if soup.title else None)
        og_img = pick("og:image", "twitter:image")
        og_desc = pick("og:description", "description")
        return {"title": og_title, "image": og_img, "description": og_desc}
    except Exception:
        return {"title": None, "image": None, "description": None}

def enrich_with_og_concurrent(products: List[Product], max_items: int) -> List[Product]:
    """Concurrently fetch OG data for up to max_items products for speed."""
    idxs = list(range(min(max_items, len(products))))
    with ThreadPoolExecutor(max_workers=min(8, len(idxs))) as ex:
        future_map = {
            ex.submit(fetch_og_metadata, products[i].url): i
            for i in idxs if products[i].url
        }
        for fut in as_completed(future_map):
            i = future_map[fut]
            try:
                og = fut.result()
                products[i].image = og.get("image")
                products[i].og_title = og.get("title")
            except Exception:
                pass
    return products

# ---------- Product extraction ----------

RETAILER_DOMAINS = {
    "amazon.de","mediamarkt.de","saturn.de","idealo.de","thomann.de","otto.de",
    "notebooksbilliger.de","cyberport.de","alternate.de","conrad.de",
    "sennheiser-hearing.com","sony.de","bose.de","apple.com","teufel.de",
    "euronics.de","saturn.at","mediamarkt.at"
}

LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
PRICE_RE = re.compile(r"(?:€\s?\d{1,4}(?:[\.,]\d{2})?|\d{1,4}(?:[\.,]\d{2})?\s?€|\bEUR\b\s?\d{1,4})")

def parse_products_from_json_block(text: str) -> List[Product]:
    """
    Preferred: parse a fenced ```json block containing {"products":[...]}
    """
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in blocks:
        try:
            data = json.loads(block)
            if isinstance(data, dict) and isinstance(data.get("products"), list):
                return _products_from_dict_list(data["products"])
        except Exception:
            continue
    return []

def parse_products_from_any_json(text: str) -> List[Product]:
    """
    Fallback: find any inline {...} chunk that looks like it has "products":[]
    """
    matches = re.findall(r"(\{[^{}]{0,2000}\"products\"\s*:\s*\[.*?\][^{}]{0,2000}\})", text, flags=re.DOTALL)
    for m in matches:
        try:
            data = json.loads(m)
            if isinstance(data, dict) and isinstance(data.get("products"), list):
                return _products_from_dict_list(data["products"])
        except Exception:
            continue
    return []

def _products_from_dict_list(items: List[dict]) -> List[Product]:
    out: List[Product] = []
    for p in items:
        out.append(Product(
            name=p.get("name") or "",
            price=p.get("price") if isinstance(p.get("price"), (int, float)) else _to_float_price(str(p.get("price")) if p.get("price") is not None else None),
            currency=p.get("currency"),
            review_score=float(p["review_score"]) if p.get("review_score") is not None else None,
            availability=p.get("availability"),
            shop=p.get("shop"),
            url=p.get("url"),
            specs=p.get("specs", {}) if isinstance(p.get("specs"), dict) else {}
        ))
    return out

def parse_products_from_markdown(markdown: str) -> List[Product]:
    """
    Heuristic extraction from the LLM's Markdown when JSON is missing.
    - Use headings as product names (### Product).
    - Collect retailer links under/near that section.
    - Sniff price on the same or next lines.
    """
    lines = markdown.splitlines()
    products: List[Product] = []
    current_name: Optional[str] = None

    def flush_placeholder():
        # We only add a product when we have at least a name or a retailer link
        pass

    section_start_idx = -1
    sections = []
    for i, line in enumerate(lines):
        if line.strip().startswith("### "):
            sections.append((i, line.strip()[4:].strip()))
    # Add sentinel end
    sections.append((len(lines), None))

    for idx in range(len(sections) - 1):
        start, name = sections[idx]
        end, _ = sections[idx + 1]
        if not name:
            continue
        body = "\n".join(lines[start:end])

        # find retailer links
        links = LINK_RE.findall(body)
        # keep only retailer domains
        retailer_links = []
        for text, url in links:
            dom = _domain(url).lower()
            if any(dom.endswith(d) for d in RETAILER_DOMAINS):
                retailer_links.append((text, url, dom))

        if not retailer_links and not name:
            continue

        # Extract one price near each retailer link occurrence
        # We search around the line containing the link text
        product_price = None
        product_currency = "EUR"
        product_shop = None
        product_url = None

        if retailer_links:
            # Take the first retailer link as canonical "buy" link
            text, url, dom = retailer_links[0]
            product_url = url
            product_shop = dom.split("/")[0]
            # Search for price in the whole section body near the link text
            # 1) On the same line as the link text
            found_price = None
            for ln in body.splitlines():
                if text in ln or url in ln:
                    m = PRICE_RE.search(ln)
                    if m:
                        found_price = m.group(0)
                        break
            # 2) Otherwise, take the first price in section
            if not found_price:
                m2 = PRICE_RE.search(body)
                if m2:
                    found_price = m2.group(0)
            product_price = _to_float_price(found_price) if found_price else None

        products.append(Product(
            name=name,
            price=product_price,
            currency=product_currency if product_price is not None else None,
            review_score=None,
            availability=None,
            shop=product_shop,
            url=product_url,
            specs={}
        ))

    # Deduplicate by (name, url)
    dedup = {}
    for p in products:
        key = (p.name.lower(), p.url or "")
        if key not in dedup:
            dedup[key] = p
        else:
            # Prefer the one with a price
            if dedup[key].price is None and p.price is not None:
                dedup[key] = p
    return list(dedup.values())

def products_to_dataframe(products: List[Product]) -> pd.DataFrame:
    records = []
    for p in products:
        records.append({
            "name": p.name,
            "price": p.price,
            "currency": p.currency,
            "review_score": p.review_score,
            "availability": p.availability,
            "shop": p.shop,
            "url": p.url,
        })
    return pd.DataFrame.from_records(records)

# ------- Reasoning (steps) formatting helpers -------
def _strip_headings_and_html(text: str) -> str:
    """Remove Markdown headings and HTML tags, normalize whitespace."""
    if not text:
        return ""
    t = re.sub(r"(?m)^#{1,6}\s*", "", text)     # remove markdown heading hashes
    t = re.sub(r"</?[^>]+>", " ", t)            # strip HTML tags
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _escape_md(text: str) -> str:
    """Escape markdown control chars so plain text never becomes headers/etc."""
    if not text:
        return ""
    return re.sub(r"([\\`*_{}\[\]()#+!\-])", r"\\\1", text)

def _truncate(text: str, max_len: int = 300) -> str:
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= max_len else text[:max_len].rstrip() + " …"

def _extract_price_lines(text: str, limit: int = 3) -> List[str]:
    """Find lines containing price-like patterns (€, EUR, 99.99)."""
    if not text:
        return []
    candidates = []
    for line in text.splitlines():
        if re.search(r"(\d+[.,]\d{2}\s?€)|(\bEUR\b\s?\d+)|(\b\d+[.,]\d{2}\b)", line, re.IGNORECASE):
            candidates.append(_truncate(line, 140))
        if len(candidates) >= limit:
            break
    return candidates

def _summarize_search_observation(obs: Any, max_items: int = 5) -> list[dict]:
    """
    Return a list of {title, url, domain, snippet} items with all headings/HTML stripped.
    """
    if not isinstance(obs, list):
        if isinstance(obs, dict) and "results" in obs and isinstance(obs["results"], list):
            obs = obs["results"]
        else:
            # Fallback single item
            return [{"title": _truncate(_strip_headings_and_html(str(obs)), 80), "url": "", "domain": "", "snippet": ""}]

    items = []
    for item in obs[:max_items]:
        url = item.get("url", "") or ""
        title_raw = item.get("title") or url
        snippet_raw = item.get("content", "")

        title = _escape_md(_strip_headings_and_html(_truncate(title_raw, 100)))
        snippet = _escape_md(_strip_headings_and_html(_truncate(snippet_raw, 140)))
        dom = _domain(url)

        items.append({"title": title, "url": url, "domain": dom, "snippet": snippet})
    return items


def _summarize_scrape_observation(obs: Any) -> str:
    """
    Clean and summarize scrape output — no giant headers.
    Strips Markdown headings and HTML tags, shows short snippet + price lines.
    """
    if obs is None:
        return "_No readable content_"

    text = str(obs)

    # Remove Markdown headers (###, ##, etc.) and HTML tags (<h1>, <p>, etc.)
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)  # remove Markdown heading hashes
    text = re.sub(r"</?[^>]+>", " ", text)       # strip HTML tags
    text = re.sub(r"\s+", " ", text).strip()     # normalize whitespace

    # Short snippet and price detection
    snippet = _truncate(text, 240)
    prices = _extract_price_lines(text)

    out = []
    if snippet:
        out.append(f"📰 **Snippet:** {snippet}")
    if prices:
        out.append("💰 **Price lines:**\n" + "\n".join([f"- {p}" for p in prices]))

    return "\n\n".join(out) if out else "_No readable content_"


# ================
# LangChain Tools
# ================

@tool
def scrape_web_page(url: str) -> str:
    """
    Scrapes the content of a single web page given its URL.
    Returns a trimmed text (max ~3000 chars). Use after search on promising URLs.
    """
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        content = " ".join([doc.page_content for doc in docs])
        return content[:3000]
    except Exception as e:
        print(f"Error scraping page {url}: {e}")
        return f"Error scraping page: {e}"

# ================
# Cached resources
# ================

@st.cache_resource(show_spinner=False)
def get_search_tool(max_results: int = 3):
    return TavilySearchResults(max_results=max_results)

@st.cache_resource(show_spinner=False)
def get_llm(model_name: str = "gemini-2.5-flash", temperature: float = 0.5):
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

# ==================
# Agent Construction
# ==================

def create_shopping_agent(model_name: str = "gemini-2.5-flash"):
    llm = get_llm(model_name=model_name)
    search_tool = get_search_tool(max_results=3)
    tools = [search_tool, scrape_web_page]
    llm_with_tools = llm.bind_tools(tools)

    json_example_escaped = (
        "{{\n"
        '  "products": [\n'
        "    {{\n"
        '      "name": "string",\n'
        '      "price": 199.99,\n'
        '      "currency": "EUR",\n'
        '      "review_score": 4.5,\n'
        '      "availability": "In Stock",\n'
        '      "shop": "MediaMarkt",\n'
        '      "url": "https://...",\n'
        '      "specs": {{"key": "value"}}\n'
        "    }}\n"
        "  ]\n"
        "}}\n"
    )

    system_instructions = (
        "You are an expert product researcher. Your goal is to find the best products for the user based on their query.\n\n"
        "Your process:\n"
        "1) Use the tavily_search_results_json tool to find reputable sources (German/EU focus).\n"
        "2) Use scrape_web_page on top URLs to extract exact product names, specs, prices, and purchase links.\n"
        "3) Propose 2–3 top contenders with concise pros/cons and clean, canonical links only (no tracking).\n"
        "4) IMPORTANT: In addition to the Markdown answer, ALSO output a machine-readable fenced JSON block at the end.\n"
        "   Start the block with ```json and end it with ```.\n"
        "   The JSON must look like:\n"
        f"{json_example_escaped}\n"
        "Do not invent URLs or prices; use null if unknown.\n"
        'Add at the very end of the Markdown: "Prices and availability are subject to change."'
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instructions),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_tool_messages(x["intermediate_steps"]),
        }
        | prompt
        | llm_with_tools
        | ToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True
    )
    return agent_executor

# ===============
# Streamlit UI
# ===============

st.set_page_config(page_title="Smart Shopper Agent", page_icon="🛍️", layout="wide")
st.title("🛍️ Smart Shopper Agent")
st.markdown("Your personal AI assistant to find the best products for you!")

with st.sidebar:
    model_choice = st.selectbox(
        "Model",
        ["gemini-2.5-flash (fast)", "gemini-2.5-pro (best)"],
        index=0
    )
    model_name = "gemini-2.5-flash" if "flash" in model_choice else "gemini-2.5-pro"
    show_previews = st.checkbox("Show product preview cards (slower)", value=False)
    st.caption("Tip: Turn previews on after first results to keep it snappy.")

query = st.text_input(
    "What are you looking for today?",
    placeholder="e.g., best noise-canceling headphones under 200€"
)

col_a, col_b = st.columns([1, 3])
with col_a:
    do_search = st.button("🔎 Search", type="primary", use_container_width=True)
with col_b:
    st.caption("The agent only runs when you press **Search**. Filters & toggles won’t re-run it.")

with st.expander("Advanced options"):
    show_reasoning = st.checkbox("Show how I picked these (tools & sources)", value=False)

# Session state to avoid re-running the agent on each widget change
if "agent_markdown" not in st.session_state:
    st.session_state.agent_markdown = ""
if "products" not in st.session_state:
    st.session_state.products = []
if "intermediate_steps" not in st.session_state:
    st.session_state.intermediate_steps = []

if do_search and query.strip():
    with st.spinner("🤖 Searching the web and analyzing product pages..."):
        try:
            shopping_agent = create_shopping_agent(model_name=model_name)

            prompt_for_agent = f"""
Find the best product for the user based on their query: '{query}'.

Steps:
1) Use your search tool to find recent reviews, comparisons, and product listings from reputable shops in the EU/DE region.
2) Use your scraping tool on the most promising URLs to find exact details.
3) Identify 2–3 top contenders that fit the user's intent.
4) For each contender:
   - Summarize key features and notable pros/cons (concise).
   - Provide 2–3 direct purchase links from reputable shops (e.g., Amazon.de, MediaMarkt.de, Saturn.de, Idealo.de, Thomann.de).
   - Include the current price and availability from the scraped page, if visible.
   - Use clean, canonical URLs (no tracking parameters).

Final Formatting (Markdown):
- **🏆 Top Picks:** short bullet list (product name + one-line reason).
- **✨ Product Details:** use the given template per product.
- **🤔 Final Verdict:** one paragraph.
- Add: "Prices and availability are subject to change." at the end.

Additionally, include the requested JSON block with 'products' as specified in the system message.
""".strip()

            result = shopping_agent.invoke({"input": prompt_for_agent})
            agent_markdown = result.get("output", "")

            # 1) Try proper fenced JSON
            products = parse_products_from_json_block(agent_markdown)
            # 2) Try any inline JSON with "products"
            if not products:
                products = parse_products_from_any_json(agent_markdown)
            # 3) Heuristic parse from Markdown (headings + retailer links)
            if not products:
                products = parse_products_from_markdown(agent_markdown)

            # store in session for fast UI updates
            st.session_state.agent_markdown = agent_markdown
            st.session_state.products = products or []
            st.session_state.intermediate_steps = result.get("intermediate_steps", [])

        except Exception as e:
            st.error(f"An error occurred: {e}")

# ----- Render main results from session -----
if st.session_state.agent_markdown:
    # Remove fenced JSON blocks before showing to user
    cleaned_markdown = re.sub(
        r"```json\s*{.*?}\s*```",
        "",
        st.session_state.agent_markdown,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    st.markdown(cleaned_markdown)


products = st.session_state.products

# Optional OG cards (concurrent, limited first)
if products and show_previews:
    st.subheader("🔎 Quick Preview Cards")
    enrich_with_og_concurrent(products, max_items=min(6, len(products)))

    cols_per_row = 3
    rows = (len(products) + cols_per_row - 1) // cols_per_row
    for r in range(rows):
        cols = st.columns(cols_per_row)
        for i in range(cols_per_row):
            idx = r * cols_per_row + i
            if idx >= len(products):
                break
            p = products[idx]
            with cols[i]:
                if p.image:
                    st.image(p.image, use_container_width=True)
                st.markdown(f"**{p.og_title or p.name}**")
                price_str = f"{p.price:.2f} {p.currency or ''}".strip() if p.price is not None else "Price not shown"
                rs = f"{p.review_score}/5" if p.review_score is not None else "N/A"
                st.caption(
                    f"Shop: {p.shop or '—'} • Price: {price_str} • Reviews: {rs} • Availability: {p.availability or '—'}"
                )
                if p.url:
                    try:
                        st.link_button("Open product", p.url, use_container_width=True)
                    except Exception:
                        st.markdown(f"[Open product]({p.url})")

    if len(products) > 6 and st.button("Enrich all previews (slower)"):
        with st.spinner("Fetching previews…"):
            enrich_with_og_concurrent(products, max_items=len(products))
        st.rerun()

# Comparison table + filters (no agent re-run)
if products:
    st.subheader("📊 Compare & Filter")
    df = products_to_dataframe(products)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        # Default 0.0 so unrated items aren’t dropped silently
        min_score = st.slider("Min review score", 0.0, 5.0, 0.0, 0.1)
    with c2:
        max_price = st.number_input("Max price (€)", value=10000.0, step=10.0)
    with c3:
        availability_filter = st.selectbox(
            "Availability",
            options=["Any", "In Stock", "Out of Stock", "Preorder"],
            index=0
        )
    with c4:
        include_unrated = st.checkbox("Include unrated", value=True)

    filtered = df.copy()
    filtered["review_score"] = pd.to_numeric(filtered["review_score"], errors="coerce")
    filtered["price"] = pd.to_numeric(filtered["price"], errors="coerce")

    score_mask = (filtered["review_score"] >= min_score)
    if include_unrated:
        score_mask = score_mask | (filtered["review_score"].isna())

    price_mask = (filtered["price"].fillna(float("inf")) <= max_price)
    avail_mask = True
    if availability_filter != "Any":
        avail_mask = (filtered["availability"].fillna("") == availability_filter)

    filtered = filtered[score_mask & price_mask & avail_mask]

    if filtered.empty:
        st.info("No products match the current filters. Try lowering Min review score or enabling 'Include unrated'.")
    st.dataframe(
        filtered[["name", "price", "currency", "review_score", "availability", "shop", "url"]],
        use_container_width=True
    )

    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Download CSV",
            filtered.to_csv(index=False),
            "smartshopper_comparison.csv"
        )
    with e2:
        md_cols = ["name", "price", "currency", "review_score", "shop", "url"]
        md_header = "| " + " | ".join(md_cols) + " |\n|" + " | ".join(["---"] * len(md_cols)) + "|\n"
        md_rows = "\n".join(
            "| " + " | ".join(str(x) if x is not None else "" for x in row) + " |"
            for row in filtered[md_cols].values
        )
        st.download_button(
            "Copy Markdown Table",
            md_header + md_rows,
            "smartshopper_table.md"
        )


# ---------- Improved reasoning reveal ----------
def render_reasoning(steps: List[Any]):
    if not steps:
        st.info("No intermediate steps available.")
        return

    for idx, step in enumerate(steps, start=1):
        try:
            action, observation = step
        except Exception:
            st.code(_truncate(str(step), 1200), language="text")
            continue

        tool_name = getattr(action, "tool", None) or "tool"
        raw_input = getattr(action, "tool_input", None)

        with st.container(border=True):
            st.markdown(f"**Step {idx}:** `{tool_name}`")

            if tool_name.lower() in ("tavily_search_results_json", "tavily_search_results", "search"):
                st.caption("Search results summary")
                st.markdown(_summarize_search_observation(observation))

            elif tool_name.lower() in ("scrape_web_page", "scrape", "web_scrape"):
                url_val = None
                if isinstance(raw_input, dict):
                    url_val = raw_input.get("url")
                elif isinstance(raw_input, str):
                    try:
                        parsed = json.loads(raw_input)
                        if isinstance(parsed, dict):
                            url_val = parsed.get("url")
                    except Exception:
                        pass
                if url_val:
                    st.write(f"**URL:** [{url_val}]({url_val})  — *{_domain(url_val)}*")
                st.caption("Scrape summary")
                st.markdown(_summarize_scrape_observation(observation))
            else:
                st.caption("Tool output")
                st.code(_truncate(str(observation), 1200), language="text")

            with st.expander("Show tool input / raw payload"):
                if raw_input is not None:
                    try:
                        st.code(json.dumps(raw_input, indent=2) if not isinstance(raw_input, str) else raw_input, language="json")
                    except Exception:
                        st.code(str(raw_input), language="text")
                st.caption("Observation (raw)")
                try:
                    st.code(
                        json.dumps(observation, indent=2) if isinstance(observation, (dict, list)) else _truncate(str(observation), 4000),
                        language="json" if isinstance(observation, (dict, list)) else "text"
                    )
                except Exception:
                    st.code(_truncate(str(observation), 1200), language="text")

# Render reasoning if toggled
if st.session_state.intermediate_steps and show_reasoning:
    st.subheader("🔍 How I picked these (tools & sources)")
    render_reasoning(st.session_state.intermediate_steps)

st.caption("Tip: Press **Search** to run the agent. Filters & toggles won’t re-run it. Previews are optional for speed.")
