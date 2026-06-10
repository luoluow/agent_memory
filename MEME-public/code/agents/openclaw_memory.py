"""
OpenClaw memory system for MeME evaluation.

Reimplements the *default* built-in memory of OpenClaw's `memory-core` extension
(github.com/openclaw/openclaw), evaluated with the same harness, answerer, and
judge as every other MeME approach.

OpenClaw memory-core, default config:
  - Storage: plain-markdown notes in the workspace. `MEMORY.md` (long-term,
    evergreen) plus APPEND-ONLY `memory/YYYY-MM-DD.md` daily notes. Backed by a
    SQLite index over the NOTE files (not raw transcripts — session-transcript
    indexing is the off-by-default `experimental.sessionMemory` flag).
  - Ingest: the "pre-compaction memory flush" — a silent LLM turn that extracts
    durable facts and APPENDS them to the dated note (flush-plan.ts). MEMORY.md is
    read-only during flush; dreaming (the opt-in promotion-into-MEMORY.md pass) is
    OFF by default, so MEMORY.md stays empty here.
  - Retrieval: the `memory_search` tool — HYBRID search: embedding vector search
    + BM25, weighted-merged, top-K, clamped to a char budget (memory-search.md).
    Temporal decay and MMR are off by default.
  - Forgetting: none structural. No tombstones; notes are append-only, so a
    changed/deleted fact leaves the old line in place and the new line appended.

So OpenClaw is RAG-over-append-only-distilled-notes with hybrid semantic+keyword
retrieval — the first *semantic* (vector) retriever in this benchmark lineup, and
distinct from Hermes (keyword FTS over raw transcripts) and auto-memory (read the
whole overwrite-curated file).

Fidelity notes (consistent with the auto_memory / hermes adapters):
  - The flush LLM pass is skipped for filler sessions (no durable facts; a real
    flush would extract ~nothing). The index covers notes, not raw transcripts, so
    filler is simply never written — matching the default (no sessionMemory).
  - Vector half uses local nomic-embed-text via Ollama (off the answerer's quota);
    BM25 via SQLite FTS5. Flush extraction + answering use claude-code, as for the
    other adapters.
"""

import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from typing import Dict, List, Optional

from openai import OpenAI

from agents.base import BaseMemorySystem

EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = os.environ.get("OMNI_OLLAMA_BASE_URL", "http://localhost:11434/v1")
SEARCH_TOP_K = 12
SEARCH_CHAR_BUDGET = 6000
VECTOR_WEIGHT = 0.5
BM25_WEIGHT = 0.5

_embed_client: Optional[OpenAI] = None


def _client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
    return _embed_client


def _embed(text: str, prefix: str) -> List[float]:
    resp = _client().embeddings.create(model=EMBED_MODEL, input=prefix + text)
    return resp.data[0].embedding


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


FLUSH_SYSTEM_PROMPT = """\
You are OpenClaw's pre-compaction memory flush. Read ONE conversation session and \
extract the DURABLE facts worth remembering long-term — anything about the user's \
life, work, preferences, possessions, relationships, routines, decisions, and \
their CHANGES.

Rules:
- Write one durable fact per line, as a terse note: "- <fact>". Names/values VERBATIM.
- Capture state CHANGES and REMOVALS explicitly as their own note when the user \
states them ("switched X to Y", "no longer does Z", "cancelled/ended Z", "broke up").
- Capture conditional dependencies ("my Y depends on my X", "if X changes Y becomes Z").
- These notes are APPENDED to a dated log — do not restate unchanged old facts, \
only what THIS session establishes or changes.
- Skip pure small talk. If nothing durable, output exactly: (nothing)

Output ONLY the note lines (or "(nothing)"). No JSON, no preamble, no headers."""


# Dreaming "deep phase" promotion/consolidation: distill the append-only daily notes
# into a deduplicated, RESOLVED long-term MEMORY.md (the opt-in consolidation that, in
# real OpenClaw, promotes high-value notes into MEMORY.md). Surfaced always-on at answer
# time alongside search — the analog of an always-loaded curated memory file.
DREAM_SYSTEM_PROMPT = """\
You are OpenClaw's dreaming consolidation. You are given dated memory notes accumulated
over time (append-only, so they contain stale values, changes, and removals all mixed
together). Distill them into a clean, durable long-term memory.

Rules:
- One fact per line: "- <entity>: <current value>". Names/values VERBATIM.
- RESOLVE changes: for a fact that changed, keep only the LATEST value (use the dates).
- For things that accumulate (hobbies, skills, appointments), keep the FULL current list.
- If the notes say something was removed/cancelled/ended with no replacement, record it
  as "- <entity>: removed/no longer applies" (do NOT carry the old value forward).
- Drop pure trivia. Be complete but compact.

Output ONLY the consolidated note lines. No preamble, no headers."""


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
    cmd = ["claude", "-p", "--output-format", "text", "--no-session-persistence"]
    if "/" in model:
        cmd.extend(["--model", model.split("/", 1)[1]])
    cmd.extend(["--system-prompt", system])
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                            timeout=timeout)
    out = result.stdout.strip()
    if result.returncode != 0 and not out:
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}): {result.stderr.strip()[:300]}")
    return out


class OpenClawMemory(BaseMemorySystem):
    """MeME agent reimplementing OpenClaw memory-core default (append notes + hybrid memory_search)."""

    def __init__(self, model: str = "claude-code",
                 base_tmp_dir: Optional[str] = None, dreaming: bool = False):
        self.model = model
        self.base_tmp_dir = base_tmp_dir or tempfile.gettempdir()
        self.dreaming = dreaming
        self._dir: Optional[str] = None
        self._db: Optional[sqlite3.Connection] = None
        self._fts5 = True
        # in-memory vector store parallel to the FTS rows: id -> (text, date, embedding)
        self._vecs: Dict[int, dict] = {}
        self._next_id = 0
        self._memory_md = ""   # consolidated long-term memory (populated by dreaming)
        self._last_retrieved_context = ""

    # ---- storage helpers ----
    def _notes_dir(self) -> str:
        return os.path.join(self._dir, "memory")

    def _note_path(self, date: str) -> str:
        safe = re.sub(r"[^0-9A-Za-z_-]", "_", date or "undated") or "undated"
        return os.path.join(self._notes_dir(), f"{safe}.md")

    def _open_db(self):
        self._db = sqlite3.connect(os.path.join(self._dir, "index.db"))
        try:
            self._db.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(body)")
            self._fts5 = True
        except sqlite3.OperationalError:
            self._db.execute("CREATE TABLE chunks_fts (rowid INTEGER PRIMARY KEY, body TEXT)")
            self._fts5 = False
        self._db.commit()

    def _index_chunk(self, text: str, date: str):
        """Add one durable-fact note line to the hybrid index (BM25 row + vector)."""
        text = text.strip()
        if not text:
            return
        cid = self._next_id
        self._next_id += 1
        if self._fts5:
            self._db.execute("INSERT INTO chunks_fts(rowid, body) VALUES (?, ?)", (cid, text))
        else:
            self._db.execute("INSERT INTO chunks_fts(rowid, body) VALUES (?, ?)", (cid, text))
        self._db.commit()
        try:
            emb = _embed(text, "search_document: ")
        except Exception:
            emb = None
        self._vecs[cid] = {"text": text, "date": date, "emb": emb}

    # ---- hybrid memory_search ----
    def _bm25_scores(self, question: str) -> Dict[int, float]:
        terms = [t for t in re.findall(r"[A-Za-z0-9]+", question.lower()) if len(t) >= 3]
        if not terms:
            return {}
        out: Dict[int, float] = {}
        if self._fts5:
            q = " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))
            try:
                cur = self._db.execute(
                    "SELECT rowid, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? "
                    "ORDER BY bm25(chunks_fts) LIMIT 50", (q,))
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                rows = []
            # bm25() returns lower = better (negative-ish). Convert to higher = better.
            for rid, score in rows:
                out[rid] = -float(score)
        else:
            clause = " OR ".join(["body LIKE ?"] * len(terms))
            cur = self._db.execute(
                f"SELECT rowid FROM chunks_fts WHERE {clause}", [f"%{t}%" for t in terms])
            for (rid,) in cur.fetchall():
                out[rid] = 1.0
        return out

    @staticmethod
    def _normalize(scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def _memory_search(self, question: str) -> str:
        """OpenClaw memory_search: hybrid vector + BM25, weighted-merge, top-K, clamp."""
        if not self._vecs:
            return ""
        # vector half
        vec_scores: Dict[int, float] = {}
        try:
            qemb = _embed(question, "search_query: ")
            for cid, rec in self._vecs.items():
                if rec["emb"] is not None:
                    vec_scores[cid] = _cosine(qemb, rec["emb"])
        except Exception:
            vec_scores = {}
        bm25_scores = self._bm25_scores(question)

        vn = self._normalize(vec_scores)
        bn = self._normalize(bm25_scores)
        merged: Dict[int, float] = {}
        for cid in set(vn) | set(bn):
            merged[cid] = VECTOR_WEIGHT * vn.get(cid, 0.0) + BM25_WEIGHT * bn.get(cid, 0.0)
        ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:SEARCH_TOP_K]

        out, used = [], 0
        for cid, _ in ranked:
            rec = self._vecs[cid]
            line = f"[{rec['date']}] {rec['text']}" if rec.get("date") else rec["text"]
            if used + len(line) > SEARCH_CHAR_BUDGET and out:
                break
            out.append(line)
            used += len(line)
        return "\n".join(out)

    # ---- BaseMemorySystem ----
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
        self._dir = os.path.join(self.base_tmp_dir, f"meme_openclaw_{os.getpid()}_{ts}")
        os.makedirs(self._notes_dir(), exist_ok=True)
        open(os.path.join(self._dir, "MEMORY.md"), "w").close()  # evergreen, empty (dreaming off)
        self._vecs = {}
        self._next_id = 0
        self._memory_md = ""
        self._open_db()
        self._last_retrieved_context = ""

    def ingest_session(self, session: dict) -> dict:
        if self._dir is None:
            self.reset()

        # Default config indexes NOTES, not raw transcripts; filler yields no durable
        # flush, so it is neither flushed nor indexed (matches sessionMemory=off).
        if session.get("type") == "filler":
            return {"skipped": True, "reason": "filler (no durable flush)",
                    "chunks": len(self._vecs),
                    "token_usage": {"input_tokens": 0, "output_tokens": 0}}

        date = (session.get("timestamp") or "").split("T")[0].split(" ")[0]
        body = _format_session(session)
        try:
            raw = _call_claude(body, FLUSH_SYSTEM_PROMPT, self.model)
        except Exception as e:
            return {"error": str(e), "chunks": len(self._vecs),
                    "token_usage": {"input_tokens": 0, "output_tokens": 0}}

        lines = []
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln or ln.lower() == "(nothing)":
                continue
            ln = re.sub(r"^[-*]\s*", "", ln).strip()
            if ln:
                lines.append(ln)

        # APPEND to the dated note (never overwrite). Index the whole flush as ONE
        # contiguous chunk (matches OpenClaw's multi-line snippet retrieval; keeps a
        # session's related facts together so aggregation isn't fragmented).
        if lines:
            block = "\n".join(f"- {l}" for l in lines)
            with open(self._note_path(date), "a") as f:
                f.write(block + "\n")
            self._index_chunk(block, date)

        return {"flushed": len(lines), "chunks": len(self._vecs),
                "token_usage": {"input_tokens": 0, "output_tokens": 0}}

    def finalize_ingest(self):
        """Dreaming consolidation: distill append-only notes into a resolved MEMORY.md.

        OpenClaw's opt-in 'dreaming' promotes/consolidates high-value notes into the
        long-term MEMORY.md. Runs once per phase (after all sessions are ingested).
        No-op when dreaming is disabled (the default config)."""
        if not self.dreaming:
            return
        nd = self._notes_dir()
        note_blocks = []
        if os.path.isdir(nd):
            for fn in sorted(os.listdir(nd)):
                if fn.endswith(".md"):
                    with open(os.path.join(nd, fn)) as f:
                        body = f.read().strip()
                    if body:
                        note_blocks.append(f"[{fn[:-3]}]\n{body}")
        notes = "\n".join(note_blocks)
        if not notes.strip():
            return
        try:
            consolidated = _call_claude(notes, DREAM_SYSTEM_PROMPT, self.model)
        except Exception:
            return
        consolidated = consolidated.strip()
        if consolidated:
            self._memory_md = consolidated
            with open(os.path.join(self._dir, "MEMORY.md"), "w") as f:
                f.write(consolidated + "\n")

    def retrieve(self, question: str) -> str:
        """Consolidated MEMORY.md (always-on, when dreaming populated it) + one hybrid
        memory_search over the indexed notes (query budget = 1)."""
        search = self._memory_search(question)
        parts = []
        if self.dreaming and self._memory_md.strip():
            parts.append("# Long-term memory (MEMORY.md)\n" + self._memory_md)
        if search:
            parts.append("# memory_search results (notes)\n" + search)
        self._last_retrieved_context = "\n\n".join(parts) if parts else "(no memory)"
        return self._last_retrieved_context

    def get_memory_snapshot(self) -> dict:
        parts = []
        if self.dreaming and self._memory_md.strip():
            parts.append("## MEMORY.md (consolidated)\n" + self._memory_md)
        nd = self._notes_dir()
        if os.path.isdir(nd):
            for fn in sorted(os.listdir(nd)):
                if fn.endswith(".md"):
                    with open(os.path.join(nd, fn)) as f:
                        body = f.read().strip()
                    if body:
                        parts.append(f"## memory/{fn}\n{body}")
        return {"text": "\n\n".join(parts) or "(no memory)"}

    def get_retrieved_context(self) -> str:
        return self._last_retrieved_context
