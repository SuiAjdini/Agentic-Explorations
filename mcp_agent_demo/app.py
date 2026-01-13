from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from google import genai

from mcp_client import McpConnection, McpTool

load_dotenv()

st.set_page_config(page_title="MCP + Gemini Agent", layout="wide")


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set in .env")
    return genai.Client(api_key=api_key)


SYSTEM_INSTRUCTIONS = """You are an AI agent connected to an MCP tool server having access to a 'notes' directory. 

# STRATEGY (CRITICAL):
1. **Always List First**: If the user asks about a specific project or topic, call 'list_notes' first to see correct filenames.
2. **Read Full Files**: Do not rely on 'search_notes' for comprehensive answers. Once you identify the relevant file from the list, use 'read_note' to get the full content.
3. **Search fallback**: Only use 'search_notes' if you are looking for a keyword across many unknown files. Keep search queries short (1-2 words).

# TOOLS:
You must be explicit about when you need to use a tool.
When you want to use a tool, respond ONLY with a JSON object in this exact format:
{
  "action": "tool",
  "tool_name": "...",
  "tool_args": { ... }
}

When you have enough information to answer, respond ONLY with:
{
  "action": "final",
  "answer": "..."
}

Do not output anything else besides JSON.
"""

def clean_json_response(text: str) -> str:
    """Removes Markdown code block formatting from Gemini responses."""
    text = text.strip()
    # Remove opening ```json or ```
    if text.startswith("```"):
        lines = text.splitlines()
        # If the first line is just ``` or ```json, remove it
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        # If the last line is ```, remove it
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()

async def agent_turn(
    gemini: genai.Client,
    model: str,
    user_text: str,
    tools: List[McpTool],
    mcp: McpConnection,
    max_steps: int = 6,
) -> str:
    tool_catalog = [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]

    prompt = (
        SYSTEM_INSTRUCTIONS
        + "\n\nAvailable MCP tools (JSON):\n"
        + json.dumps(tool_catalog, ensure_ascii=False)
        + "\n\nUser request:\n"
        + user_text
        + "\n"
    )

    for i in range(max_steps):
        resp = gemini.models.generate_content(model=model, contents=prompt)
        raw = (resp.text or "").strip()
        cleaned_raw = clean_json_response(raw)
        try:
            obj = json.loads(cleaned_raw)
        except Exception:
            # Force back to JSON-only format
            prompt += "\n\nYou MUST output only valid JSON. Try again.\n"
            continue

        if obj.get("action") == "final":
            return obj.get("answer", "")

        if obj.get("action") == "tool":
            tool_name = obj.get("tool_name")
            tool_args = obj.get("tool_args") or {}
            if not isinstance(tool_args, dict):
                tool_args = {}

            tool_result = await mcp.call_tool(tool_name, tool_args)

            prompt += (
                f"\n\nTool call requested:\n{raw}\n"
                f"\nTool result ({tool_name}):\n{tool_result}\n"
                "\nNow either call another tool or return final. JSON only.\n"
            )
            continue

        prompt += "\n\nUnknown action. Output JSON with action=tool or action=final.\n"

    return "I hit the tool-call step limit. Please refine the request."


@st.cache_resource
def get_event_loop():
    # Streamlit can re-run; keep a stable loop for async calls.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop) 
    return loop


async def ensure_mcp_connected(server_path: str) -> McpConnection:
    mcp = McpConnection()
    await mcp.connect_stdio(server_path)
    return mcp


st.title("MCP + Gemini + Streamlit Agent")
st.caption("Demonstrates MCP discovery (tools/resources/prompts) and tool execution with Gemini reasoning.")


with st.sidebar:
    st.subheader("Connection")
    server_path = st.text_input("MCP server script path", value="servers/notes_server.py")
    model_name = st.text_input("Gemini model", value="gemini-2.5-flash") 

    connect = st.button("Connect")

if "mcp" not in st.session_state:
    st.session_state.mcp = None
    st.session_state.tools = []
    st.session_state.resources = []
    st.session_state.prompts = []
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of (role, text)

loop = get_event_loop()

if connect:
    mcp = loop.run_until_complete(ensure_mcp_connected(server_path))
    st.session_state.mcp = mcp
    st.session_state.tools = loop.run_until_complete(mcp.list_tools())
    st.session_state.resources = loop.run_until_complete(mcp.list_resources())
    st.session_state.prompts = loop.run_until_complete(mcp.list_prompts())
    st.success("Connected to MCP server.")


tab_chat, tab_explorer = st.tabs(["Chat", "MCP Explorer"])

with tab_explorer:
    st.subheader("Tools")
    st.json([t.__dict__ for t in st.session_state.tools])

    st.subheader("Resources")
    st.json(st.session_state.resources)

    st.subheader("Prompts")
    st.json(st.session_state.prompts)

with tab_chat:
    for role, text in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(text)

    user_msg = st.chat_input("Ask something that may require reading/searching notes…")
    if user_msg:
        st.session_state.chat.append(("user", user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)

        if not st.session_state.mcp:
            assistant_text = "Not connected to an MCP server yet. Connect in the sidebar first."
        else:
            gemini = get_gemini_client()
            assistant_text = loop.run_until_complete(
                agent_turn(
                    gemini=gemini,
                    model=model_name,
                    user_text=user_msg,
                    tools=st.session_state.tools,
                    mcp=st.session_state.mcp,
                )
            )

        st.session_state.chat.append(("assistant", assistant_text))
        with st.chat_message("assistant"):
            st.markdown(assistant_text)
