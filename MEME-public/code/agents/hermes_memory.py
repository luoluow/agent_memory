"""
Hermes Agent memory system for MeME evaluation.

Reimplements the *built-in* memory architecture of NousResearch's Hermes Agent
(https://github.com/nousresearch/hermes-agent), evaluated with the same harness,
answerer, and judge as every other MeME approach so the numbers are comparable.

Hermes' built-in memory has two layers:

1. Bounded, curated files in ~/.hermes/memories/ (a "frozen snapshot" injected
   at session start):
     - USER.md   — notes about the user        (1,375 char / ~500 tok cap)
     - MEMORY.md — notes about the world/work   (2,200 char / ~800 tok cap)
   The tight budget *forces curation*: the agent must consolidate/merge entries
   to stay under the cap. Here, each evidence session triggers one `claude -p`
   pass that rewrites both files within budget (hard-truncated as a backstop).

2. A session archive — every session logged to SQLite with FTS5 full-text
   search, queried at answer time via the `session_search` tool. This is the
   episodic layer that recalls verbatim detail the bounded files had to drop.

retrieve() (query budget = 1) = the frozen USER.md + MEMORY.md snapshot PLUS one
session_search over the question. answer_question() uses the base class.

Fidelity notes:
  - Like the auto_memory baseline, the curation `claude -p` pass is skipped for
    filler sessions (they carry no tracked evidence facts). ALL sessions —
    filler included — are still indexed into the FTS archive, since searching
    over distractor turns is exactly what Hermes' session_search does.
  - The memory tool's add/replace/remove ops are modeled as a full-file rewrite
    under the real char caps; what the eval tests is the bounded curated content
    and the FTS recall layer, both preserved here.
"""

import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from typing import Dict, List, Optional

from agents.base import BaseMemorySystem

# Real Hermes built-in budgets (docs/user-guide/features/memory).
USER_MD_LIMIT = 1375     # ~500 tokens
MEMORY_MD_LIMIT = 2200   # ~800 tokens

# session_search shaping
SEARCH_TOP_K = 6
SEARCH_CHAR_BUDGET = 6000  # cap total search context so verbatim facts survive but prompt stays sane

_STOPWORDS = {
    "the", "and", "for", "with", "what", "when", "where", "which", "who", "whom",
    "are", "was", "were", "has", "have", "had", "did", "does", "you", "your",
    "their", "they", "this", "that", "these", "those", "from", "about", "into",
    "over", "than", "then", "there", "here", "all", "any", "can", "could",
    "would", "should", "will", "now", "current", "currently", "still", "list",
    "tell", "give", "say", "said", "name", "much", "many", "user", "users",
}


INGEST_SYSTEM_PROMPT = f"""\
You are the Hermes Agent built-in memory system. You curate two bounded memory \
files and must keep each STRICTLY within its character budget — the budget is the \
point: it forces you to keep only durable, generalizable facts and to \
consolidate/merge overlapping entries to make room for new ones.

## The two files
- USER.md   — notes about the user (who they are, their profile, possessions, \
relationships, routines, preferences, current scalar facts). HARD CAP: \
{USER_MD_LIMIT} characters.
- MEMORY.md — notes about the world/work/project and anything not strictly about \
the user. HARD CAP: {MEMORY_MD_LIMIT} characters.

## Rules
- Write one compact fact per line ("- key: value"). Be terse; names and values \
VERBATIM, no prose.
- Read the NEW session and update the files: add new facts, REPLACE values that \
changed, and REMOVE facts the user explicitly ended/cancelled/deleted.
- When a file would exceed its cap, consolidate: merge related lines and drop the \
least durable detail. Never exceed the cap.
- Keep facts that future questions might need: current values, what changed, what \
was removed.

## Output
Output ONLY valid JSON, no markdown fences, no explanation:
{{"user_md": "<full new contents of USER.md>", "memory_md": "<full new contents of MEMORY.md>"}}
"""


def _format_session(session: dict) -> str:
    ts = session.get("timestamp", "")
    parts = [f"[Session: {ts}]"] if ts else ["[Session]"]
    for turn in session.get("conversation", []):
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _call_claude(prompt: str, system: str, model: str = "claude-code",
                 timeout: int = 180) -> str:
    import subprocess
    cmd = ["claude", "-p", "--output-format", "text", "--no-session-persistence"]
    if "/" in model:
        cmd.extend(["--model", model.split("/", 1)[1]])
    cmd.extend(["--system-prompt", system])
    result = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, timeout=timeout,
    )
    output = result.stdout.strip()
    if result.returncode != 0 and not output:
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}): {result.stderr.strip()[:300]}"
        )
    return output


def _parse_files(raw_output: str) -> Optional[Dict[str, str]]:
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Salvage the largest balanced {...} span.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(parsed, dict):
        return None
    return {
        "user_md": (parsed.get("user_md") or "").strip(),
        "memory_md": (parsed.get("memory_md") or "").strip(),
    }


def _enforce_cap(content: str, cap: int) -> str:
    """Hard-truncate to the char cap on a line boundary (Hermes never exceeds the cap)."""
    if len(content) <= cap:
        return content
    truncated = content[:cap]
    nl = truncated.rfind("\n")
    if nl > cap // 2:
        truncated = truncated[:nl]
    return truncated.rstrip()


def _fts_query(question: str) -> str:
    """Build a safe FTS5 MATCH query: salient tokens OR'd together, each quoted."""
    tokens = re.findall(r"[A-Za-z0-9]+", question.lower())
    terms = []
    seen = set()
    for t in tokens:
        if len(t) < 3 or t in _STOPWORDS or t in seen:
            continue
        seen.add(t)
        terms.append(f'"{t}"')
    return " OR ".join(terms)


class HermesMemory(BaseMemorySystem):
    """MeME agent reimplementing Hermes Agent's built-in memory (bounded files + FTS session archive)."""

    def __init__(self, model: str = "claude-code",
                 base_tmp_dir: Optional[str] = None):
        self.model = model
        self.base_tmp_dir = base_tmp_dir or tempfile.gettempdir()
        self._dir: Optional[str] = None
        self._db: Optional[sqlite3.Connection] = None
        self._fts5 = True
        self._last_retrieved_context: str = ""
        self._answer_token_usage: Dict = {"input_tokens": 0, "output_tokens": 0}

    # ---- file helpers ----
    def _user_path(self) -> str:
        return os.path.join(self._dir, "USER.md")

    def _memory_path(self) -> str:
        return os.path.join(self._dir, "MEMORY.md")

    def _read(self, path: str) -> str:
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            return ""

    def _snapshot_text(self) -> str:
        user = self._read(self._user_path())
        mem = self._read(self._memory_path())
        parts = []
        if user:
            parts.append("## USER.md\n" + user)
        if mem:
            parts.append("## MEMORY.md\n" + mem)
        return "\n\n".join(parts)

    # ---- session archive (SQLite + FTS5) ----
    def _open_db(self):
        self._db = sqlite3.connect(os.path.join(self._dir, "state.db"))
        try:
            self._db.execute(
                "CREATE VIRTUAL TABLE sessions USING fts5(session_id, timestamp, body)"
            )
            self._fts5 = True
        except sqlite3.OperationalError:
            # FTS5 not compiled in — fall back to a plain table + LIKE search.
            self._db.execute(
                "CREATE TABLE sessions (session_id TEXT, timestamp TEXT, body TEXT)"
            )
            self._fts5 = False
        self._db.commit()

    def _index_session(self, session: dict, body: str):
        self._db.execute(
            "INSERT INTO sessions(session_id, timestamp, body) VALUES (?, ?, ?)",
            (session.get("session_id", ""), session.get("timestamp", ""), body),
        )
        self._db.commit()

    def _session_search(self, question: str) -> str:
        """Hermes' session_search: FTS5 ranked recall over the raw session archive."""
        rows: List[str] = []
        if self._fts5:
            q = _fts_query(question)
            if not q:
                return ""
            try:
                cur = self._db.execute(
                    "SELECT body FROM sessions WHERE sessions MATCH ? ORDER BY rank LIMIT ?",
                    (q, SEARCH_TOP_K),
                )
                rows = [r[0] for r in cur.fetchall()]
            except sqlite3.OperationalError:
                rows = []
        else:
            terms = [t for t in re.findall(r"[A-Za-z0-9]+", question.lower())
                     if len(t) >= 3 and t not in _STOPWORDS]
            if not terms:
                return ""
            clause = " OR ".join(["body LIKE ?"] * len(terms))
            cur = self._db.execute(
                f"SELECT body FROM sessions WHERE {clause} LIMIT ?",
                [f"%{t}%" for t in terms] + [SEARCH_TOP_K],
            )
            rows = [r[0] for r in cur.fetchall()]

        out, used = [], 0
        for body in rows:
            if used + len(body) > SEARCH_CHAR_BUDGET and out:
                break
            out.append(body)
            used += len(body)
        return "\n\n---\n\n".join(out)

    # ---- BaseMemorySystem interface ----
    def reset(self):
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
            self._db = None
        if self._dir and os.path.isdir(self._dir):
            shutil.rmtree(self._dir, ignore_errors=True)
        ts = int(time.time() * 1000)
        self._dir = os.path.join(self.base_tmp_dir, f"meme_hermes_{os.getpid()}_{ts}")
        os.makedirs(self._dir, exist_ok=True)
        open(self._user_path(), "w").close()
        open(self._memory_path(), "w").close()
        self._open_db()
        self._last_retrieved_context = ""
        self._answer_token_usage = {"input_tokens": 0, "output_tokens": 0}

    def ingest_session(self, session: dict) -> dict:
        if self._dir is None:
            self.reset()

        body = _format_session(session)
        # Episodic layer: archive EVERY session (filler included) for session_search.
        self._index_session(session, body)

        # Curation layer: skip the LLM pass for filler (no tracked facts), like auto_memory.
        if session.get("type") == "filler":
            return {
                "skipped": True,
                "reason": "filler (archived to session_search, not curated)",
                "user_entries": self._read(self._user_path()).count("\n") + 1 if self._read(self._user_path()) else 0,
                "memory_entries": self._read(self._memory_path()).count("\n") + 1 if self._read(self._memory_path()) else 0,
                "token_usage": {"input_tokens": 0, "output_tokens": 0},
            }

        user_md = self._read(self._user_path())
        memory_md = self._read(self._memory_path())
        prompt = (
            f"## Current USER.md ({len(user_md)}/{USER_MD_LIMIT} chars)\n{user_md or '(empty)'}\n\n"
            f"## Current MEMORY.md ({len(memory_md)}/{MEMORY_MD_LIMIT} chars)\n{memory_md or '(empty)'}\n\n"
            f"## New session\n{body}\n\n"
            f"Update both files within their caps and return the JSON."
        )

        try:
            raw = _call_claude(prompt, INGEST_SYSTEM_PROMPT, self.model)
        except Exception as e:
            return {"error": str(e), "memory_entries": 0,
                    "token_usage": {"input_tokens": 0, "output_tokens": 0}}

        parsed = _parse_files(raw)
        if parsed is not None:
            new_user = _enforce_cap(parsed["user_md"], USER_MD_LIMIT)
            new_mem = _enforce_cap(parsed["memory_md"], MEMORY_MD_LIMIT)
            with open(self._user_path(), "w") as f:
                f.write(new_user + ("\n" if new_user else ""))
            with open(self._memory_path(), "w") as f:
                f.write(new_mem + ("\n" if new_mem else ""))

        u, m = self._read(self._user_path()), self._read(self._memory_path())
        return {
            "user_chars": len(u),
            "memory_chars": len(m),
            "user_entries": u.count("\n") + 1 if u else 0,
            "memory_entries": m.count("\n") + 1 if m else 0,
            "parse_ok": parsed is not None,
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
        }

    def retrieve(self, question: str) -> str:
        """Frozen USER.md + MEMORY.md snapshot PLUS one session_search (query budget = 1)."""
        snapshot = self._snapshot_text()
        search = self._session_search(question)

        parts = []
        if snapshot:
            parts.append("# Curated memory (USER.md / MEMORY.md)\n" + snapshot)
        if search:
            parts.append("# session_search results (past conversations)\n" + search)
        context = "\n\n".join(parts) if parts else "(no memory)"
        self._last_retrieved_context = context
        return context

    def get_memory_snapshot(self) -> dict:
        return {"text": self._snapshot_text() or "(no memory)"}

    def get_retrieved_context(self) -> str:
        return self._last_retrieved_context
