import os
import json
import time
import ast
import operator as op

import streamlit as st
from dotenv import load_dotenv

from google import genai

from langfuse import get_client, observe, propagate_attributes
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor


# ----------------------------
# Setup
# ----------------------------
load_dotenv()

# 1) Instrument Google GenAI SDK with OpenTelemetry -> Langfuse
# This makes Gemini calls show up automatically in Langfuse.
GoogleGenAIInstrumentor().instrument()

# 2) Langfuse client
langfuse = get_client()
if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    assert langfuse.auth_check(), "Langfuse auth failed — check LANGFUSE_PUBLIC_KEY/SECRET_KEY"

# 3) Gemini client
# The SDK can read GEMINI_API_KEY from env automatically.

client = genai.Client()

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ----------------------------
# A safe calculator tool
# ----------------------------
_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod,
}

def _safe_eval_expr(expr: str) -> float:
    """
    Evaluate a simple arithmetic expression safely.
    Supports: +, -, *, /, **, %, parentheses, unary minus.
    """
    def _eval(node):
        if isinstance(node, ast.Num):  # py<3.8
            return node.n
        if isinstance(node, ast.Constant):  # py>=3.8
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numbers are allowed.")
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed.")
            return _ALLOWED_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _ALLOWED_OPERATORS:
                raise ValueError("Unary operator not allowed.")
            return _ALLOWED_OPERATORS[type(node.op)](_eval(node.operand))
        raise ValueError("Invalid expression.")

    parsed = ast.parse(expr, mode="eval")
    return _eval(parsed.body)


@observe(name="tool_calculator")
def tool_calculator(expression: str) -> str:
    try:
        result = _safe_eval_expr(expression)
        return str(result)
    except Exception as e:
        return f"Calculator error: {e}"


# ----------------------------
# Agent logic
# ----------------------------
PLANNER_SYSTEM = """You are a small assistant that can either:
1) call a calculator tool, or
2) answer directly.

You MUST respond with valid JSON only (no markdown, no code fences) in this schema:
{
  "action": "tool" | "final",
  "tool_name": "calculator" | null,
  "tool_input": "string" | null,
  "answer": "string" | null
}

Rules:
- If the user asks for math or an arithmetic result, use action="tool" with tool_name="calculator"
  and tool_input containing ONLY the math expression (no extra text).
- Otherwise, action="final" with a helpful answer.
"""

@observe(name="gemini_generate")
def gemini_generate(contents: str) -> str:
    resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
    )
    return resp.text or ""


@observe(name="agent_turn")
def run_agent_turn(user_text: str) -> str:
    # 1) Planner step (decide tool vs final)
    planner_input = f"{PLANNER_SYSTEM}\n\nUser: {user_text}\nJSON:"
    raw = gemini_generate(planner_input)

    # Robust JSON parse (handles minor whitespace)
    try:
        plan = json.loads(raw.strip())
    except json.JSONDecodeError:
        # Fallback: if model misbehaves, answer directly
        return f"(Planner JSON parse failed) Here’s my best answer:\n\n{gemini_generate(user_text)}"

    # 2) Tool call if requested
    if plan.get("action") == "tool" and plan.get("tool_name") == "calculator":
        expr = (plan.get("tool_input") or "").strip()
        tool_out = tool_calculator(expr)

        # 3) Final response grounded on tool output
        final_prompt = (
            "You are the assistant.\n"
            f"User asked: {user_text}\n"
            f"Calculator result: {tool_out}\n\n"
            "Respond to the user in 1-3 sentences, friendly and clear."
        )
        return gemini_generate(final_prompt)

    # 4) Direct final
    return plan.get("answer") or gemini_generate(user_text)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Gemini Agent + Langfuse Demo", page_icon="🧠")
st.title("🧠 Simple Gemini Agent — traced in Langfuse")

with st.expander("Setup checklist", expanded=False):
    st.markdown(
        "- Set `GEMINI_API_KEY` and `LANGFUSE_*` env vars\n"
        "- Run the app and ask a math question (e.g. `12*7 + 3`)\n"
        "- Open Langfuse → Traces to see the agent steps + Gemini calls"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Ask something… (try math to trigger the tool)")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Attach consistent trace attributes (user/session) for this turn
    session_id = st.session_state.get("session_id")
    if not session_id:
        session_id = f"streamlit-{int(time.time())}"
        st.session_state.session_id = session_id

    with propagate_attributes(
        user_id="linkedin-demo",
        session_id=session_id,
        tags=["streamlit", "gemini", "agent-demo"],
        metadata={"app": "langfuse-gemini-streamlit-agent"},
        version="0.1.0",
    ):
        answer = run_agent_turn(user_text)
        # Store turn-level input/output on the current trace
        langfuse.update_current_trace(input=user_text, output=answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    langfuse.flush()
