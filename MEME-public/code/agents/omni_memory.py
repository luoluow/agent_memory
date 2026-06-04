"""
OmniMemory — high-accuracy memory system using a local LLM (Ollama) for all
memory management operations.

Architecture (see external_docs/OmniMemory_design.md):

  Local LLM (Ollama) handles:
    EXTRACT    — per-session entity extraction (qwen2.5:7b, fast)
    RELATE     — edge inference for new entities (qwen2.5:7b)
    COMPRESS   — filler session summaries (qwen2.5:7b)
    VERIFY     — post-phase full-text reconciliation (qwen2.5:14b)

  Cloud LLM handles:
    answer_question() — the MeME unified answer prompt (unchanged)

Storage layout per episode:
  raw/session_NNN.txt  — verbatim session archive (never modified)
  state.json           — current entity values (scalar and list)
  history.jsonl        — append-only revision log
  deletions.json       — explicit deletion ledger
  pages/{entity}.md    — KG-style wiki pages with [[links]]
  summaries.jsonl      — compressed filler context

Key innovations over prior approaches:
  1. EXTRACT receives current values (not just names) → better change detection
  2. VERIFY reads raw session archive (not memory files) → catches cascade misses
  3. Separate deletions.json → Del/Abs signals always at top of context
  4. Multi-valued entity lists → fixes Agg overwrites
  5. Local LLM → zero memory-op cost, true background concurrency
"""

import json
import os
import re
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Set

from openai import OpenAI

from agents.base import BaseMemorySystem


# ---------------------------------------------------------------------------
# Local LLM client  (Ollama OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434/v1"
EXTRACT_MODEL   = "qwen2.5:7b"    # fast, used for EXTRACT / RELATE / COMPRESS
VERIFY_MODEL    = "qwen2.5:14b"   # stronger reasoning for VERIFY

_local_client: Optional[OpenAI] = None


def _get_local_client() -> OpenAI:
    global _local_client
    if _local_client is None:
        _local_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    return _local_client


def _call_local(prompt: str, system: str, model: str = EXTRACT_MODEL,
                max_tokens: int = 1024, temperature: float = 0.0) -> str:
    client = _get_local_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _extract_json(text: str) -> str:
    """Extract JSON object from text that may contain prose or code fences."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return text


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """\
You extract entity updates from a conversation session into structured memory.

ENTITY TYPES:
- scalar: single value (medication, partner, workplace, vehicle, project_name)
- list: multiple simultaneous values (hobbies, skills, appointments, languages, activities)

RULES:
- Use snake_case entity names. Match existing names exactly (shown below).
- current_value: VERBATIM from session text — do not paraphrase.
- type "list": if user says "I also do X" or "I do X and Y" → append. If "I stopped X" → remove item.
- type "scalar": if user says "I now do X instead" → replace value.
- deleted: true ONLY if user explicitly removes/cancels/discontinues an entity entirely.
- Include ONLY entities that actually changed this session.
- If nothing changed: {"updates": []}

Output ONLY valid JSON starting with {
{"updates": [{"entity": "name", "type": "scalar|list", "new_value": "string or [list]", "deleted": false, "timestamp": "YYYY/MM/DD"}]}
"""

RELATE_SYSTEM = """\
Identify typed directed relationships between new entities and existing entities.

Edge types: treats, affects, proximate_to, destination, requires, implies,
            part_of, managed_by, lives_with, related_to, works_at, owns

Return only meaningful relationships (not trivial ones).
Output ONLY valid JSON starting with {
{"edges": [{"from": "entity_a", "to": "entity_b", "type": "edge_type"}]}
"""

COMPRESS_SYSTEM = """\
Write 1-2 sentences summarizing the key facts from these conversation sessions.
Be specific: names, values, dates. Focus on what was established or changed.
Output ONLY plain text — no JSON, no headers.
"""

VERIFY_SYSTEM = """\
You are a memory auditor. Given complete session transcripts and the current entity \
state, find discrepancies.

Check for:
1. CORRECTIONS — entity values that changed in the sessions but are wrong or missing \
   in the current state. Include the EXACT VERBATIM value from the session text.
2. MISSED_DELETIONS — entities explicitly removed, cancelled, or discontinued in the \
   sessions but not marked deleted in the current state.
3. LIST_CORRECTIONS — list entities where the current list is incomplete or has stale items.

Be conservative: only flag clear discrepancies with evidence from the text.
Do NOT speculate or infer values not explicitly stated.

Output ONLY valid JSON starting with {
{
  "corrections": [
    {"entity": "name", "correct_value": "verbatim value or [list]",
     "was": "old value", "evidence": "short quote from session"}
  ],
  "missed_deletions": [
    {"entity": "name", "was": "old value", "evidence": "short quote"}
  ],
  "list_corrections": [
    {"entity": "name", "correct_list": ["item1", "item2"],
     "evidence": "short quote"}
  ]
}
"""


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _raw_dir(memory_dir: str) -> str:
    d = os.path.join(memory_dir, "raw")
    os.makedirs(d, exist_ok=True)
    return d


def _pages_dir(memory_dir: str) -> str:
    d = os.path.join(memory_dir, "pages")
    os.makedirs(d, exist_ok=True)
    return d


def _state_path(memory_dir: str) -> str:
    return os.path.join(memory_dir, "state.json")


def _history_path(memory_dir: str) -> str:
    return os.path.join(memory_dir, "history.jsonl")


def _deletions_path(memory_dir: str) -> str:
    return os.path.join(memory_dir, "deletions.json")


def _summaries_path(memory_dir: str) -> str:
    return os.path.join(memory_dir, "summaries.jsonl")


# State

def _load_state(memory_dir: str) -> Dict[str, dict]:
    path = _state_path(memory_dir)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_state(memory_dir: str, state: Dict[str, dict],
                lock: Optional[threading.Lock] = None) -> None:
    path = _state_path(memory_dir)
    if lock:
        with lock:
            with open(path, "w") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
    else:
        with open(path, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)


# Deletions

def _load_deletions(memory_dir: str) -> Dict[str, dict]:
    path = _deletions_path(memory_dir)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_deletions(memory_dir: str, deletions: Dict[str, dict]) -> None:
    with open(_deletions_path(memory_dir), "w") as f:
        json.dump(deletions, f, ensure_ascii=False, indent=2)


# History

def _append_history(memory_dir: str, event: dict,
                    lock: Optional[threading.Lock] = None) -> None:
    path = _history_path(memory_dir)
    line = json.dumps(event, ensure_ascii=False) + "\n"
    if lock:
        with lock:
            with open(path, "a") as f:
                f.write(line)
    else:
        with open(path, "a") as f:
            f.write(line)


def _read_history(memory_dir: str, n: int = 40) -> List[dict]:
    path = _history_path(memory_dir)
    if not os.path.exists(path):
        return []
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    return events[-n:]


# Summaries

def _append_summary(memory_dir: str, summary: str, ts: str) -> None:
    with open(_summaries_path(memory_dir), "a") as f:
        f.write(json.dumps({"ts": ts, "summary": summary}) + "\n")


def _read_summaries(memory_dir: str, n: int = 4) -> List[dict]:
    path = _summaries_path(memory_dir)
    if not os.path.exists(path):
        return []
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    return items[-n:]


# Pages

def _page_path(memory_dir: str, entity: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", entity)
    return os.path.join(_pages_dir(memory_dir), f"{safe}.md")


def _load_page(memory_dir: str, entity: str) -> str:
    path = _page_path(memory_dir, entity)
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read()


def _parse_links(content: str) -> List[str]:
    return re.findall(r"\[\[([^\]\|]+)\]\]", content)


def _write_page(memory_dir: str, entity: str, state: Dict[str, dict],
                edges: Optional[List[dict]] = None) -> None:
    info = state.get(entity, {})
    val = info.get("current_value", "?")
    ts  = info.get("last_updated", "?")
    src = info.get("update_source", "extract")

    existing = _load_page(memory_dir, entity)
    existing_edges: List[str] = []
    history_block = ""

    if existing:
        m = re.search(r"## Relationships\n(.*?)(?=\n##|\Z)", existing, re.DOTALL)
        if m:
            existing_edges = [l.strip() for l in m.group(1).splitlines() if l.strip()]
        m = re.search(r"## History\n(.*)", existing, re.DOTALL)
        if m:
            history_block = m.group(1).strip()

    # Add new history row
    if isinstance(val, list):
        val_str = ", ".join(val)
    else:
        val_str = str(val) if val is not None else "DELETED"

    new_row = f"| {ts} | {val_str} | {src} |"
    if not history_block:
        history_block = "| Date | Value | Source |\n|------|-------|--------|\n" + new_row
    else:
        history_block += f"\n{new_row}"

    # Merge new edges
    if edges:
        for e in edges:
            if e["from"] == entity:
                line = f"- {e['type']} → [[{e['to']}]]"
                if line not in existing_edges:
                    existing_edges.append(line)
            elif e["to"] == entity:
                line = f"- {e['type']} ← [[{e['from']}]]"
                if line not in existing_edges:
                    existing_edges.append(line)

    rel_block = "\n".join(existing_edges) if existing_edges else "(none)"
    tag = " [verified]" if src == "verified" else ""
    display_val = f"[{', '.join(val)}]" if isinstance(val, list) else (str(val) if val else "DELETED")

    content = f"""\
---
entity: {entity}
current_value: {json.dumps(val)}
last_updated: {ts}
update_source: {src}
---

# {entity}

**Current**: {display_val}{tag}

## Relationships
{rel_block}

## History
{history_block}
"""
    with open(_page_path(memory_dir, entity), "w") as f:
        f.write(content)


def _format_session(session: dict) -> str:
    ts = session.get("timestamp", "")
    parts = [f"[Session: {ts}]"] if ts else ["[Session]"]
    for turn in session.get("conversation", []):
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _format_value(val: Any) -> str:
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val) if val is not None else "?"


def _assemble_context(memory_dir: str) -> str:
    state     = _load_state(memory_dir)
    deletions = _load_deletions(memory_dir)
    history   = _read_history(memory_dir, 40)
    summaries = _read_summaries(memory_dir, 4)

    parts = []

    # 1. Deletions — always at top, explicit and prominent
    if deletions:
        lines = [
            "## ⚠ DELETED — No current value",
            "(For these: answer 'was deleted' / 'no longer applicable' / 'I don't know')\n",
        ]
        for entity, info in sorted(deletions.items()):
            lines.append(
                f"- **{entity}**: was {info.get('was', '?')} "
                f"(removed {info.get('when', '?')}). "
                f"{info.get('tombstone', 'No replacement.')}"
            )
        parts.append("\n".join(lines))

    # 2. Current entity state
    if state:
        lines = ["## Current Entity State\n"]
        for entity, info in sorted(state.items()):
            val = _format_value(info.get("current_value"))
            ts  = info.get("last_updated", "?")
            src = info.get("update_source", "")
            tag = " [verified]" if src == "verified" else ""
            lines.append(f"- **{entity}**: {val} (as of {ts}){tag}")
        parts.append("\n".join(lines))

    # 3. Entity relationship pages (for Agg)
    pages_d = _pages_dir(memory_dir)
    page_contents = []
    for fname in sorted(os.listdir(pages_d)):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(pages_d, fname)) as f:
            content = f.read().strip()
        if content:
            page_contents.append(content)
    if page_contents:
        parts.append("## Entity Relationships\n\n" + "\n\n---\n\n".join(page_contents))

    # 4. Revision history (for Tr)
    if history:
        lines = ["## Revision History\n"]
        for ev in history:
            e   = ev.get("entity", "?")
            old = ev.get("old")
            new = ev.get("new")
            ts  = ev.get("ts", "?")
            src = ev.get("source", "")
            tag = " [verified]" if src == "verified" else ""
            if ev.get("deleted"):
                lines.append(f"[{ts}] {e}: DELETED (was: {_format_value(old)}){tag}")
            elif old is None:
                lines.append(f"[{ts}] {e}: → {_format_value(new)}{tag}")
            else:
                lines.append(f"[{ts}] {e}: {_format_value(old)} → {_format_value(new)}{tag}")
        parts.append("\n".join(lines))

    # 5. Recent session summaries (for context fallback)
    if summaries:
        lines = ["## Recent Context\n"]
        for s in summaries:
            lines.append(f"  [{s.get('ts','')}] {s.get('summary','')}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else "(no memory)"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class OmniMemory(BaseMemorySystem):
    """
    OmniMemory: high-accuracy memory using local LLM (Ollama) for memory ops.

    Memory management (local Ollama):
      EXTRACT    — entity extraction per evidence session  (qwen2.5:7b)
      RELATE     — edge inference for new entities         (qwen2.5:7b)
      COMPRESS   — filler session summarization            (qwen2.5:7b)
      VERIFY     — post-phase full-text reconciliation     (qwen2.5:14b)

    Answer generation (cloud LLM, unchanged from MeME framework):
      retrieve() → assemble_context() → unified_llm answer prompt
    """

    HOT_K = 5  # compress filler buffer every K sessions

    def __init__(self, model: str = "claude-code",
                 extract_model: str = EXTRACT_MODEL,
                 verify_model: str = VERIFY_MODEL,
                 base_tmp_dir: Optional[str] = None):
        self.model         = model          # cloud model for answers (unused here)
        self.extract_model = extract_model
        self.verify_model  = verify_model
        self.base_tmp_dir  = base_tmp_dir or tempfile.gettempdir()

        self._memory_dir: Optional[str] = None
        self._state_lock   = threading.Lock()
        self._history_lock = threading.Lock()
        self._bg_threads:  List[threading.Thread] = []

        self._session_counter   = 0
        self._phase_sessions:   List[dict] = []  # evidence sessions this phase
        self._filler_buffer:    List[dict] = []  # filler sessions for compression
        self._last_retrieved:   str = ""

    def reset(self):
        if self._memory_dir and os.path.isdir(self._memory_dir):
            shutil.rmtree(self._memory_dir, ignore_errors=True)
        ts = int(time.time() * 1000)
        self._memory_dir = os.path.join(
            self.base_tmp_dir, f"meme_omni_{os.getpid()}_{ts}"
        )
        os.makedirs(self._memory_dir, exist_ok=True)
        _raw_dir(self._memory_dir)
        _pages_dir(self._memory_dir)
        self._bg_threads      = []
        self._session_counter = 0
        self._phase_sessions  = []
        self._filler_buffer   = []
        self._last_retrieved  = ""

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest_session(self, session: dict) -> dict:
        if self._memory_dir is None:
            self.reset()

        self._session_counter += 1
        session_text = _format_session(session)

        # Archive raw text (instant, no LLM)
        raw_path = os.path.join(
            _raw_dir(self._memory_dir),
            f"session_{self._session_counter:03d}.txt",
        )
        with open(raw_path, "w") as f:
            f.write(session_text)

        if session.get("type") == "filler":
            self._filler_buffer.append(session)
            if len(self._filler_buffer) >= self.HOT_K:
                self._compress_filler()
            return {"skipped": True, "reason": "filler session",
                    "token_usage": {"input_tokens": 0, "output_tokens": 0}}

        # Evidence session — EXTRACT in background thread
        self._phase_sessions.append(session)

        t = threading.Thread(
            target=self._extract_and_update,
            args=(session, session_text),
            daemon=True,
        )
        t.start()
        self._bg_threads.append(t)

        return {"session_archived": True, "extract_started": True,
                "token_usage": {"input_tokens": 0, "output_tokens": 0}}

    def _extract_and_update(self, session: dict, session_text: str) -> None:
        """Background: EXTRACT entity updates from one evidence session."""
        try:
            with self._state_lock:
                state = _load_state(self._memory_dir)

            # Build compact state summary for EXTRACT context
            state_summary = "\n".join(
                f'  "{k}": {json.dumps(v.get("current_value"))} '
                f'(type={v.get("type","scalar")}, as of {v.get("last_updated","?")})'
                for k, v in sorted(state.items())
            ) or "  (no entities yet)"

            prompt = (
                f"Known entities:\n{state_summary}\n\n"
                f"Session to process:\n{session_text}"
            )

            raw = _call_local(prompt, EXTRACT_SYSTEM, model=self.extract_model)
            parsed = json.loads(_extract_json(raw))
            updates = parsed.get("updates", [])

        except Exception:
            return

        if not updates:
            return

        ts = session.get("timestamp", "")[:10]
        new_entities = []

        with self._state_lock:
            state = _load_state(self._memory_dir)
            deletions = _load_deletions(self._memory_dir)

            for upd in updates:
                entity   = re.sub(r"\s+", "_", (upd.get("entity") or "").strip().lower())
                new_val  = upd.get("new_value")
                deleted  = bool(upd.get("deleted", False))
                upd_ts   = (upd.get("timestamp") or ts)[:10]
                etype    = upd.get("type", "scalar")

                if not entity:
                    continue

                old_info = state.get(entity, {})
                old_val  = old_info.get("current_value")
                is_new   = entity not in state

                if deleted:
                    was = old_val if old_val is not None else old_info.get("current_value", "?")
                    deletions[entity] = {
                        "was": was,
                        "when": upd_ts,
                        "tombstone": f"Explicitly removed as of {upd_ts}. No replacement.",
                        "source": "extract",
                    }
                    state.pop(entity, None)
                else:
                    if etype == "list" and isinstance(new_val, list):
                        existing_list = old_val if isinstance(old_val, list) else []
                        merged = list(dict.fromkeys(existing_list + new_val))
                        new_val = merged
                    state[entity] = {
                        "current_value": new_val,
                        "type": etype,
                        "first_seen": old_info.get("first_seen", upd_ts),
                        "last_updated": upd_ts,
                        "update_source": "extract",
                    }
                    if is_new:
                        new_entities.append(entity)

                _append_history(self._memory_dir, {
                    "entity": entity, "old": old_val, "new": new_val if not deleted else None,
                    "ts": upd_ts, "deleted": deleted, "source": "extract",
                }, lock=self._history_lock)

            _save_state(self._memory_dir, state)
            _save_deletions(self._memory_dir, deletions)

            # Update entity pages
            for entity in [u.get("entity","") for u in updates if not u.get("deleted")]:
                entity = re.sub(r"\s+", "_", entity.strip().lower())
                if entity and entity in state:
                    _write_page(self._memory_dir, entity, state)

        # RELATE: infer edges for newly created entities
        if new_entities and len(state) > len(new_entities):
            self._relate_new_entities(new_entities, state)

    def _relate_new_entities(self, new_entities: List[str],
                             state: Dict[str, dict]) -> None:
        """Infer typed edges from new entities to existing ones."""
        existing = {e: state[e] for e in state if e not in new_entities}
        if not existing:
            return

        new_summary = "\n".join(
            f'  {e}: {json.dumps(state[e].get("current_value"))}'
            for e in new_entities
        )
        existing_summary = "\n".join(
            f'  {e}: {json.dumps(info.get("current_value"))}'
            for e, info in sorted(existing.items())
        )

        prompt = (
            f"New entities:\n{new_summary}\n\n"
            f"Existing entities:\n{existing_summary}"
        )

        try:
            raw = _call_local(prompt, RELATE_SYSTEM, model=self.extract_model,
                              max_tokens=512)
            parsed = json.loads(_extract_json(raw))
            edges = parsed.get("edges", [])
            with self._state_lock:
                state = _load_state(self._memory_dir)
                for entity in new_entities + [e["to"] for e in edges] + [e["from"] for e in edges]:
                    entity = re.sub(r"\s+", "_", entity.strip().lower())
                    if entity in state:
                        _write_page(self._memory_dir, entity, state, edges)
        except Exception:
            pass

    def _compress_filler(self) -> None:
        if not self._filler_buffer:
            return
        sessions_text = "\n\n---\n\n".join(
            _format_session(s) for s in self._filler_buffer
        )
        last_ts = (self._filler_buffer[-1].get("timestamp") or "")[:10]
        first_ts = (self._filler_buffer[0].get("timestamp") or "")[:10]
        try:
            summary = _call_local(
                f"Sessions from {first_ts} to {last_ts}:\n\n{sessions_text}",
                COMPRESS_SYSTEM, model=self.extract_model, max_tokens=200,
            )
            if summary:
                _append_summary(self._memory_dir, summary.strip(), last_ts)
        except Exception:
            pass
        self._filler_buffer = []

    # ------------------------------------------------------------------
    # Post-phase VERIFY
    # ------------------------------------------------------------------

    def finalize_ingest(self) -> None:
        """Join background threads, flush filler, then VERIFY phase sessions."""
        # Wait for all EXTRACT threads
        for t in self._bg_threads:
            t.join(timeout=120)
        self._bg_threads.clear()

        # Flush filler buffer
        if self._filler_buffer:
            self._compress_filler()

        if not self._phase_sessions or not self._memory_dir:
            self._phase_sessions = []
            return

        self._run_verify()
        self._phase_sessions = []

    def _run_verify(self) -> None:
        """VERIFY: re-read raw phase sessions, reconcile entity state."""
        state     = _load_state(self._memory_dir)
        deletions = _load_deletions(self._memory_dir)

        if not state and not deletions:
            return

        # Concatenate raw session texts for this phase
        raw_texts = []
        for s in self._phase_sessions:
            raw_texts.append(_format_session(s))
        sessions_block = "\n\n---\n\n".join(raw_texts)

        # Compact state summary
        state_summary = "\n".join(
            f'  "{k}": {json.dumps(v.get("current_value"))} '
            f'(type={v.get("type","scalar")}, updated={v.get("last_updated","?")})'
            for k, v in sorted(state.items())
        )
        deletions_summary = "\n".join(
            f'  "{k}": deleted (was {json.dumps(info.get("was"))})'
            for k, info in sorted(deletions.items())
        ) or "  (none)"

        prompt = (
            f"## Session transcripts\n\n{sessions_block}\n\n"
            f"## Current entity state\n{state_summary}\n\n"
            f"## Current deletions\n{deletions_summary}"
        )

        try:
            raw = _call_local(prompt, VERIFY_SYSTEM, model=self.verify_model,
                              max_tokens=1024, temperature=0.0)
            parsed = json.loads(_extract_json(raw))
        except Exception as e:
            print(f"      [omni] VERIFY failed: {e}")
            return

        corrections = parsed.get("corrections", [])
        missed_dels = parsed.get("missed_deletions", [])
        list_corrs  = parsed.get("list_corrections", [])

        if not corrections and not missed_dels and not list_corrs:
            return

        # Apply corrections
        ts_now = (self._phase_sessions[-1].get("timestamp", "") or "")[:10]

        with self._state_lock:
            state     = _load_state(self._memory_dir)
            deletions = _load_deletions(self._memory_dir)

            for c in corrections:
                entity    = re.sub(r"\s+", "_", (c.get("entity") or "").strip().lower())
                correct   = c.get("correct_value")
                was       = c.get("was")
                if not entity or correct is None:
                    continue
                old_info = state.get(entity, {})
                etype = old_info.get("type", "scalar")
                state[entity] = {
                    "current_value": correct,
                    "type": etype,
                    "first_seen": old_info.get("first_seen", ts_now),
                    "last_updated": ts_now,
                    "update_source": "verified",
                }
                _append_history(self._memory_dir, {
                    "entity": entity, "old": was, "new": correct,
                    "ts": ts_now, "deleted": False, "source": "verified",
                }, lock=self._history_lock)
                _write_page(self._memory_dir, entity, state)

            for lc in list_corrs:
                entity   = re.sub(r"\s+", "_", (lc.get("entity") or "").strip().lower())
                cor_list = lc.get("correct_list", [])
                if not entity or not cor_list:
                    continue
                old_info = state.get(entity, {})
                state[entity] = {
                    "current_value": cor_list,
                    "type": "list",
                    "first_seen": old_info.get("first_seen", ts_now),
                    "last_updated": ts_now,
                    "update_source": "verified",
                }
                _append_history(self._memory_dir, {
                    "entity": entity, "old": old_info.get("current_value"), "new": cor_list,
                    "ts": ts_now, "deleted": False, "source": "verified",
                }, lock=self._history_lock)
                _write_page(self._memory_dir, entity, state)

            for d in missed_dels:
                entity = re.sub(r"\s+", "_", (d.get("entity") or "").strip().lower())
                was    = d.get("was")
                if not entity:
                    continue
                old_val = state.pop(entity, {}).get("current_value", was)
                deletions[entity] = {
                    "was": old_val,
                    "when": ts_now,
                    "tombstone": f"Explicitly removed as of {ts_now}. No replacement.",
                    "source": "verified",
                }
                _append_history(self._memory_dir, {
                    "entity": entity, "old": old_val, "new": None,
                    "ts": ts_now, "deleted": True, "source": "verified",
                }, lock=self._history_lock)

            _save_state(self._memory_dir, state)
            _save_deletions(self._memory_dir, deletions)

        n = len(corrections) + len(missed_dels) + len(list_corrs)
        print(f"      [omni] VERIFY: {n} corrections applied "
              f"({len(corrections)} values, {len(missed_dels)} deletions, "
              f"{len(list_corrs)} lists)")

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> str:
        """Assemble context from all indices. No LLM call."""
        if not self._memory_dir or not os.path.isdir(self._memory_dir):
            self._last_retrieved = "(no memory)"
            return "(no memory)"
        context = _assemble_context(self._memory_dir)
        self._last_retrieved = context
        return context

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def get_memory_snapshot(self) -> dict:
        if not self._memory_dir:
            return {"text": "(no memory)"}
        return {"text": _assemble_context(self._memory_dir)}

    def get_retrieved_context(self) -> str:
        return self._last_retrieved
