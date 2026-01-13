from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import re

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("notes-server")

NOTES_DIR = Path(__file__).resolve().parent.parent / "notes"


def _safe_note_path(slug: str) -> Path:
    # Very small safety guard: only allow simple filenames (no ../ traversal)
    if not re.fullmatch(r"[a-zA-Z0-9_\-]+", slug):
        raise ValueError("Invalid note slug. Use only letters, digits, '_' and '-'.")
    return NOTES_DIR / f"{slug}.md"


@mcp.tool()
def list_notes() -> List[str]:
    """List available notes by slug (filename without extension)."""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.stem for p in NOTES_DIR.glob("*.md"))


@mcp.tool()
def read_note(slug: str) -> str:
    """Read a note by slug (e.g., 'mcp-overview')."""
    path = _safe_note_path(slug)
    if not path.exists():
        raise FileNotFoundError(f"Note not found: {slug}")
    return path.read_text(encoding="utf-8")


@mcp.tool()
def search_notes(query: str, limit: int = 5) -> List[str]:
    """Search notes for a query string and return up to `limit` short matches."""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    q = query.strip().lower()
    if not q:
        return []

    hits: List[str] = []
    for p in sorted(NOTES_DIR.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            if q in line.lower():
                hits.append(f"{p.stem}:{i}: {line.strip()[:200]}")
                if len(hits) >= limit:
                    return hits
    return hits


# Optional: expose a resource-like URI pattern (simple demonstration)
@mcp.resource("note://{slug}")
def note_resource(slug: str) -> str:
    """Resource access: read a note via a URI-like handle."""
    return read_note(slug)


# Optional: expose a reusable prompt template
@mcp.prompt("summarize_notes")
def summarize_notes_prompt(topic: str) -> str:
    """Prompt template: ask the agent to summarize notes around a topic."""
    return (
        "You are an assistant that summarizes internal notes.\n"
        f"Topic: {topic}\n"
        "Use the available tools to find relevant notes and produce:\n"
        "- A concise summary\n"
        "- 3 actionable bullets\n"
    )


if __name__ == "__main__":
    # STDIO transport by default. IMPORTANT: do not print() to stdout in MCP STDIO servers.
    mcp.run()
