import os
import re
import time
from dotenv import load_dotenv
from typing import TypedDict

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# --- 1) Environment & Config ---
load_dotenv()
st.set_page_config(page_title="Gemini + LangGraph Agent", layout="wide")
st.title("♊ Gemini Agent with Human-in-the-Loop")

st.markdown(
    """
    <style>
    .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100vw !important;
    }
    div[data-testid="column"] { min-width: 0; }
    pre { white-space: pre; overflow-x: auto; }
    </style>
    """,
    unsafe_allow_html=True,
)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ Google API Key not found. Please check your .env file.")
    st.stop()

# --- 2) Session State Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "session_v1"

def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})

# --- 3) Sidebar Controls (for demo + GIF) ---
with st.sidebar:
    st.subheader("Demo Controls")
    demo_mode = st.toggle(
        "Demo mode (force visible loops)",
        value=True,
        help="Adds stricter deterministic rules so the agent fails at least once and loops visibly."
    )
    step_delay = st.slider(
        "Step delay (seconds)",
        min_value=0.0,
        max_value=2.0,
        value=0.6,
        step=0.1,
        help="Adds a delay between steps so the loop is visible while recording a GIF."
    )
    st.divider()
    st.caption("Tip: Record the screen while running Live mode for a GIF.")

# --- 4) Define State & Model ---
class AgentState(TypedDict, total=False):
    draft: str
    critique: str
    revision_number: int
    content_quality: str

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=api_key,
)

# --- 5) Deterministic Validator ---
def validate_tweet(text: str, demo: bool) -> tuple[bool, str]:
    t = (text or "").strip()
    if not t:
        return False, "Tweet is empty."
    if len(t) > 280:
        return False, f"Too long: {len(t)} chars. Must be <= 280."
    hashtags = re.findall(r"#\w+", t)
    if len(hashtags) < 1:
        return False, "Missing hashtag. Add at least one #tag."

    if demo:
        if len(hashtags) != 2:
            return False, f"Demo rule: must have exactly 2 hashtags (found {len(hashtags)})."
        if not re.search(r"\d", t):
            return False, "Demo rule: must include at least one number."
        if "?" not in t:
            return False, "Demo rule: must include a question mark."

    return True, ""

# --- 6) Graph Nodes ---
def writer_node(state: AgentState):
    draft = state.get("draft", "")
    critique = state.get("critique", "")
    revision_number = state.get("revision_number", 0) + 1

    if not draft:
        prompt = (
            "Write ONE viral tweet about why 'AI Agents' are the future of software.\n\n"
            "Hard requirements:\n"
            "- Under 280 chars\n"
            "- Include 2–4 hashtags\n"
            "- Be concrete (include at least one practical example)\n"
            "- Return ONLY the tweet text."
        )
    else:
        prompt = (
            "Rewrite the tweet to address the feedback.\n\n"
            f"Tweet:\n{draft}\n\n"
            f"Feedback:\n{critique}\n\n"
            "Hard requirements:\n"
            "- Under 280 chars\n"
            "- Include 2–4 hashtags\n"
            "- Be concrete (include at least one practical example)\n"
            "- Return ONLY the tweet text."
        )

    msg = llm.invoke([HumanMessage(content=prompt)])
    add_message("assistant", f"**Writer (Iteration {revision_number}):**\n\n{msg.content}")
    return {"draft": msg.content, "revision_number": revision_number}

def critique_node(state: AgentState):
    draft = state.get("draft", "")

    ok, reason = validate_tweet(draft, demo=demo_mode)
    if not ok:
        add_message("assistant", f"❌ **Critique:** Failed -> {reason}")
        return {"content_quality": "fail", "critique": reason}

    prompt = f"""
Review this tweet.

Tweet:
{draft}

Return EXACTLY one of the following (first line must match):
PASS
FAIL: <single-line reason and fix>

Constraints:
- <= 280 chars
- must include hashtags (>=1)
- should be punchy and specific (avoid generic hype)
"""
    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    first_line = raw.splitlines()[0].strip()

    if first_line == "PASS":
        add_message("assistant", "✅ **Critique:** Passed!")
        return {"content_quality": "pass", "critique": "Looks good!"}

    if first_line.startswith("FAIL:"):
        feedback = first_line[len("FAIL:"):].strip() or "Failed validation (no details)."
        add_message("assistant", f"❌ **Critique:** Failed -> {feedback}")
        return {"content_quality": "fail", "critique": feedback}

    feedback = f"Unrecognized critique format: {first_line}"
    add_message("assistant", f"❌ **Critique:** Failed -> {feedback}")
    return {"content_quality": "fail", "critique": feedback}

def human_approval_node(state: AgentState):
    return {}

# --- 7) Build Graph ---
def should_continue(state: AgentState):
    if state.get("content_quality") == "pass":
        return "human_approval"
    if state.get("revision_number", 0) >= 3:
        return "human_approval"
    return "writer"

builder = StateGraph(AgentState)
builder.add_node("writer", writer_node)
builder.add_node("critique", critique_node)
builder.add_node("human_approval", human_approval_node)

builder.add_edge(START, "writer")
builder.add_edge("writer", "critique")
builder.add_conditional_edges("critique", should_continue, ["writer", "human_approval"])
builder.add_edge("human_approval", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["human_approval"])

# --- 8) Display Existing History (chat-like) ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 9) Helpers ---
def get_thread_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}

def new_thread() -> None:
    st.session_state.messages = []
    st.session_state.thread_id = f"session_{os.urandom(4).hex()}"

def html_pre(text: str) -> str:
    safe = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        "<pre style='padding:0.75rem; border:1px solid rgba(49,51,63,0.2); "
        "border-radius:0.5rem; background:rgba(0,0,0,0.02);'>"
        f"{safe}</pre>"
    )

# --- 10) Run Buttons ---
colA, colB = st.columns(2)

with colA:
    if st.button("🚀 Start New Agent (Live)"):
        new_thread()
        thread_config = get_thread_config()
        initial_input = {"revision_number": 0, "draft": "", "critique": ""}

        # Full live conversation (writer + critique) rendered in-order
        live_conversation: list[tuple[str, str]] = []  # (speaker, content)
        first_draft_text = ""
        first_critique_text = ""

        live_box = st.container()
        left, right = live_box.columns([6, 6], gap="large")

        # LEFT: live conversation feed
        with left:
            st.markdown("### Live Conversation (Writer ↔ Critique)")
            live_header = st.empty()
            convo_container = st.container()

        # RIGHT: snapshot
        with right:
            st.markdown("### Snapshot")
            side_first_draft = st.empty()
            side_first_critique = st.empty()
            side_current = st.empty()

        flow_text = "writer → critique → (loop) → human_approval"

        def render_conversation() -> None:
            with convo_container:
                # Clear by re-rendering container content each tick
                st.markdown(f"**Flow:** `{flow_text}`")
                for speaker, content in live_conversation[-10:]:  # show last 10 turns to keep it clean
                    if speaker == "writer":
                        st.markdown("**Writer:**")
                        st.markdown(html_pre(content), unsafe_allow_html=True)
                    else:
                        st.markdown("**Critique:**")
                        st.markdown(html_pre(content), unsafe_allow_html=True)

        def render_snapshot(current_iter: int, current_quality: str) -> None:
            if first_draft_text:
                side_first_draft.markdown("**First draft (Iter 1):**")
                side_first_draft.markdown(html_pre(first_draft_text), unsafe_allow_html=True)
            else:
                side_first_draft.markdown("**First draft (Iter 1):** _pending_")

            if first_critique_text:
                side_first_critique.markdown("**First critique (Iter 1):**")
                side_first_critique.markdown(html_pre(first_critique_text), unsafe_allow_html=True)
            else:
                side_first_critique.markdown("**First critique (Iter 1):** _pending_")

            side_current.markdown(
                f"**Current iteration:** `{current_iter}`  \n"
                f"**Current quality:** `{current_quality or 'n/a'}`"
            )

        with st.status("Running graph (live)...", expanded=True) as status:
            for update in graph.stream(initial_input, thread_config, stream_mode="updates"):
                node_name = next(iter(update.keys()))

                current_state = graph.get_state(thread_config)
                values = current_state.values or {}

                rev = int(values.get("revision_number", 0) or 0)
                draft = (values.get("draft") or "").strip()
                critique = (values.get("critique") or "").strip()
                quality = values.get("content_quality", "")

                live_header.markdown(f"### Iteration {rev}   |   quality: `{quality or 'n/a'}`   |   last node: `{node_name}`")

                # Append to the live conversation ONLY when the relevant node runs
                if node_name == "writer" and draft:
                    live_conversation.append(("writer", draft))
                    if rev == 1 and not first_draft_text:
                        first_draft_text = draft

                if node_name == "critique" and critique:
                    live_conversation.append(("critique", critique))
                    if rev == 1 and not first_critique_text:
                        first_critique_text = critique

                # Render live conversation + snapshot
                render_conversation()
                render_snapshot(current_iter=rev, current_quality=quality)

                if step_delay > 0:
                    time.sleep(step_delay)

                # Stop at human approval interrupt
                current_state = graph.get_state(thread_config)
                if current_state.next and current_state.next[0] == "human_approval":
                    status.update(label="Paused for human approval.", state="complete", expanded=False)
                    break
            else:
                status.update(label="Graph completed.", state="complete", expanded=False)

with colB:
    if st.button("🧹 Clear History Only"):
        st.session_state.messages = []
        st.rerun()

# --- 11) Human Approval UI ---
thread_config = get_thread_config()
current_state = graph.get_state(thread_config)

if current_state.next and current_state.next[0] == "human_approval":
    with st.container(border=True):
        st.subheader("✋ Human Approval Required")
        final_draft = (current_state.values or {}).get("draft", "")
        st.text_area("Final Draft", value=final_draft, height=140, disabled=True)

        human_feedback = st.text_area(
            "If rejecting, add feedback:",
            placeholder="What should be improved? (tone, structure, clarity, punchiness, specificity, etc.)"
        )

        col1, col2 = st.columns(2)

        if col1.button("✅ Approve"):
            add_message("user", "👍 **Human Manager:** Approved!")
            add_message("assistant", f"🎉 **Final Output Published:**\n\n{final_draft}")

            for _ in graph.stream(None, thread_config):
                pass

            st.rerun()

        if col2.button("🔄 Reject & Retry"):
            feedback = (human_feedback or "").strip()
            if not feedback:
                feedback = "Human rejected this. Improve clarity, punchiness, and specificity."

            add_message("user", f"👎 **Human Manager:** Rejected. Feedback: {feedback}")

            graph.update_state(
                thread_config,
                {"critique": feedback, "content_quality": "fail"},
                as_node="critique",
            )

            for _ in graph.stream(None, thread_config):
                pass

            st.rerun()

# --- 12) Final Success State ---
current_state = graph.get_state(thread_config)
if not current_state.next and len(st.session_state.messages) > 0:
    st.info("Workflow Completed.")
