"""omni — thin CLI client for OmniService (invoked by Claude Code hooks).

Subcommands:
    omni serve                         start the FastAPI service
    omni mcp                           start the MCP stdio server
    omni ingest   [--ns N] [--transcript P]   push new transcript turns (Stop hook)
    omni retrieve [--ns N] [--query Q | --session-start]   fetch memory (SessionStart hook)
    omni verify   [--ns N]             force a VERIFY flush
    omni snapshot [--ns N]             print full assembled memory

Hook usage reads the hook payload JSON from stdin (transcript_path, cwd, prompt).
Namespace resolution order: --ns  >  stdin cwd  >  $CLAUDE_PROJECT_DIR  >  cwd.
"""

import argparse
import json
import os
import sys
from typing import List, Optional

import httpx

from omni import config, storage


def _base_url() -> str:
    return os.environ.get("OMNI_URL", f"http://{config.HOST}:{config.PORT}")


def _read_stdin_json() -> dict:
    if sys.stdin is None or sys.stdin.isatty():
        return {}
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _resolve_ns(arg_ns: Optional[str], hook: dict) -> str:
    return (arg_ns or hook.get("cwd")
            or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())


def _resolve_client(arg_client: Optional[str]) -> str:
    return arg_client or config.CLIENT_ID


# ---------------------------------------------------------------------------
# Transcript parsing (Claude Code JSONL)
# ---------------------------------------------------------------------------

def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    chunks.append(block["text"])
                elif "content" in block and isinstance(block["content"], str):
                    chunks.append(block["content"])
        return "\n".join(chunks)
    return ""


def parse_transcript_turns(path: str) -> List[dict]:
    """Extract ordered user/assistant text turns from a Claude Code transcript JSONL."""
    turns: List[dict] = []
    if not path or not os.path.exists(path):
        return turns
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            msg = entry.get("message") if isinstance(entry, dict) else None
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            text = _content_to_text(msg.get("content")).strip()
            if text:
                turns.append({"role": role, "content": text})
    return turns


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_serve(args):
    from omni.server import main as serve_main
    serve_main()


def cmd_mcp(args):
    from omni.mcp_server import main as mcp_main
    mcp_main()


def cmd_ingest(args):
    hook = _read_stdin_json()
    ns = _resolve_ns(args.ns, hook)
    client = _resolve_client(args.client)
    transcript = args.transcript or hook.get("transcript_path")
    if not transcript:
        print(json.dumps({"ok": False, "error": "no transcript_path"}))
        return

    all_turns = parse_transcript_turns(transcript)

    # Cursor: only ingest turns newer than what we already archived for this transcript.
    d = storage.ns_dir(client, ns)
    cursor = storage.load_cursor(d)
    key = os.path.abspath(transcript)
    already = int(cursor.get(key, 0))
    new_turns = all_turns[already:]
    if not new_turns:
        print(json.dumps({"ok": True, "ingested": 0, "client_id": client, "namespace": ns}))
        return

    ts = hook.get("timestamp", "")
    try:
        resp = httpx.post(f"{_base_url()}/ingest", json={
            "client_id": client, "namespace": ns, "turns": new_turns,
            "timestamp": ts, "source": key,
        }, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return

    cursor[key] = len(all_turns)
    storage.save_cursor(d, cursor)
    print(json.dumps({"ok": True, "ingested": len(new_turns),
                      "client_id": client, "namespace": ns}))


def cmd_retrieve(args):
    hook = _read_stdin_json()
    ns = _resolve_ns(args.ns, hook)
    client = _resolve_client(args.client)
    if args.session_start:
        mode, query = "session-start", ""
    else:
        mode = "search"
        query = args.query or hook.get("prompt", "")
    try:
        resp = httpx.post(f"{_base_url()}/retrieve", json={
            "client_id": client, "namespace": ns, "query": query, "mode": mode,
        }, timeout=30)
        resp.raise_for_status()
        context = resp.json().get("context", "")
    except Exception as e:
        # Never break the session on a memory miss.
        sys.stderr.write(f"omni retrieve failed: {e}\n")
        return

    if args.session_start and not args.raw:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": f"# Project memory (OmniService)\n\n{context}",
        }}))
    else:
        print(context)


def cmd_verify(args):
    hook = _read_stdin_json()
    ns = _resolve_ns(args.ns, hook)
    client = _resolve_client(args.client)
    resp = httpx.post(f"{_base_url()}/verify", json={"client_id": client, "namespace": ns}, timeout=300)
    print(json.dumps(resp.json()))


def cmd_snapshot(args):
    hook = _read_stdin_json()
    ns = _resolve_ns(args.ns, hook)
    client = _resolve_client(args.client)
    resp = httpx.get(f"{_base_url()}/snapshot",
                     params={"client_id": client, "namespace": ns}, timeout=30)
    print(resp.json().get("text", ""))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="omni", description="OmniService client")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("serve", help="start the FastAPI service").set_defaults(func=cmd_serve)
    sub.add_parser("mcp", help="start the MCP stdio server").set_defaults(func=cmd_mcp)

    pi = sub.add_parser("ingest", help="push new transcript turns")
    pi.add_argument("--ns")
    pi.add_argument("--client", help="client id (default: $OMNI_CLIENT_ID or 'claude-code')")
    pi.add_argument("--transcript")
    pi.set_defaults(func=cmd_ingest)

    pr = sub.add_parser("retrieve", help="fetch memory context")
    pr.add_argument("--ns")
    pr.add_argument("--client")
    pr.add_argument("--query")
    pr.add_argument("--session-start", action="store_true")
    pr.add_argument("--raw", action="store_true", help="print context only (no hook JSON)")
    pr.set_defaults(func=cmd_retrieve)

    pv = sub.add_parser("verify", help="force a VERIFY flush")
    pv.add_argument("--ns")
    pv.add_argument("--client")
    pv.set_defaults(func=cmd_verify)

    ps = sub.add_parser("snapshot", help="print full assembled memory")
    ps.add_argument("--ns")
    ps.add_argument("--client")
    ps.set_defaults(func=cmd_snapshot)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
