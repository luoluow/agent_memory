"""MCP stdio server exposing OmniService memory to Claude Code in-band.

Tools:
    memory_search(query)  context-aware retrieval the model calls when it needs facts
    memory_note(text)     deliberately commit a fact (fast path, skips waiting for ingest)

Namespace resolution: $OMNI_NAMESPACE > $CLAUDE_PROJECT_DIR > cwd.
The server is a thin client of the HTTP service (OMNI_URL or config host/port).
"""

import os

import httpx
from mcp.server.fastmcp import FastMCP

from omni import config

mcp = FastMCP("omni-memory")


def _base_url() -> str:
    return os.environ.get("OMNI_URL", f"http://{config.HOST}:{config.PORT}")


def _namespace() -> str:
    return (os.environ.get("OMNI_NAMESPACE")
            or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())


def _client_id() -> str:
    return config.CLIENT_ID


@mcp.tool()
def memory_search(query: str) -> str:
    """Search long-term project/user memory for facts relevant to the current context.

    Returns deletions, current entity state, related entity pages, and revision
    history scoped to the query. Call this whenever you need to recall a prior
    decision, value, preference, or fact that may not be in the current session.
    """
    try:
        resp = httpx.post(f"{_base_url()}/retrieve", json={
            "client_id": _client_id(), "namespace": _namespace(),
            "query": query, "mode": "search",
        }, timeout=30)
        resp.raise_for_status()
        return resp.json().get("context", "(no memory)")
    except Exception as e:
        return f"(memory unavailable: {e})"


@mcp.tool()
def memory_note(text: str) -> str:
    """Commit an important fact to long-term memory immediately.

    Use for durable facts worth remembering across sessions (decisions, user
    preferences, project constraints). The note is archived and extracted into
    structured memory in the background.
    """
    try:
        resp = httpx.post(f"{_base_url()}/ingest", json={
            "client_id": _client_id(), "namespace": _namespace(),
            "turns": [{"role": "user", "content": text}],
            "source": "memory_note",
        }, timeout=30)
        resp.raise_for_status()
        return "noted"
    except Exception as e:
        return f"(could not save note: {e})"


def main():
    mcp.run()


if __name__ == "__main__":
    main()
