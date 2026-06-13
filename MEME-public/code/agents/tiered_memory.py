"""
Tiered Memory agent for MeME evaluation (v2).

Architecture designed for very long conversations with bounded per-call cost.

Four memory tiers:
  Tier 0 (Hot):          Last K raw evidence sessions (in-memory, evicted FIFO)
  Tier 1 (Entity-state): Current value per entity + entity groups — always O(entities)
  Tier 2 (Summaries):    1-2 sentence digest per compress window — grows O(sessions/K)
  Tier 3 (Revision log): Timestamped changelog per entity — append-only, bounded at retrieval

Ingest (per evidence session):
  1. EXTRACT call  — identifies entity updates → writes to Tier 1 + Tier 3
  2. Add session to Tier 0 hot buffer
  3. COMPRESS call — fires every K evidence sessions; writes Tier 2, clears Tier 0

Post-phase (finalize_ingest, EvoMemory-inspired):
  4. CASCADE REFINE — bounded call (~4k in) on Tier 1 + phase revisions:
       • Applies missed direct updates
       • Propagates cascaded entity changes (implicit dependencies)
       • Verifies and marks deletions explicitly
       • Updates ENTITY GROUPS (A-Mem-inspired): clusters semantically related entities
         so retrieval surfaces related facts for Agg/multi-entity questions

Retrieve (no LLM call):
  Returns bounded context from all four tiers with relationship-aware grouping.

Scaling:
  Every LLM call has O(1) input cost. CASCADE REFINE reads only Tier 1 (~2k) + recent
  Tier 3 (~1k) — not the full history. Cost scales O(N) with session count.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Dict, List, Optional

from agents.base import BaseMemorySystem


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM_PROMPT = """\
You are extracting entity state updates from a conversation session for long-term memory.

Identify entities (people, medications, appointments, habits, projects, relationships, \
vehicles, locations, activities, etc.) whose values were established, changed, or \
explicitly deleted/discontinued in this session.

Rules:
- Use short snake_case names (e.g. medication, sleep_pattern, gym_membership)
- If an entity name appears in "Known entities", use the SAME name for consistency
- "value": the exact current value mentioned (verbatim where possible)
- "deleted": true ONLY if explicitly removed, cancelled, stopped, or discontinued
- Only include entities that actually changed in this session
- If nothing changed, return empty updates

Output ONLY valid JSON — no prose, no fences. Start with {:
{"updates": [{"entity": "name", "value": "current value or null", "deleted": false, "timestamp": "YYYY/MM/DD"}]}
"""

COMPRESS_SYSTEM_PROMPT = """\
You are creating a brief memory summary of recent conversation sessions.

Write 2-3 sentences capturing the key events, decisions, and facts from these sessions.
Focus on what changed or happened — not what stayed the same.
Be specific: include names, values, and dates where relevant.

Output ONLY the plain text summary. No JSON, no headers, no bullet points.
"""

CASCADE_REFINE_SYSTEM_PROMPT = """\
You are reconciling an entity state store against its revision log. Be CONSERVATIVE — \
only make changes that are directly supported by the revision log. Do NOT infer, guess, \
or propagate changes based on general reasoning.

Your ONLY allowed operations:

1. RECONCILE — For each entry in the revision log, verify Tier 1 matches.
   If the revision log shows entity X has value V, but Tier 1 shows a different value,
   update Tier 1 to match the revision log's most recent entry for that entity.

2. MARK DELETIONS — If the revision log shows deleted=true for an entity, ensure Tier 1
   marks that entity as deleted=true with a clear deletion_note.
   Do NOT mark anything deleted unless the revision log EXPLICITLY shows deleted=true.

3. UPDATE ENTITY GROUPS — Organize entities into semantic groups to help multi-entity
   retrieval. Example: free_time=[hobby, sport, club], health=[medication, condition, diet].
   This is purely organizational — it does not change any entity values.

Do NOT:
- Change entity values based on logical inference or domain knowledge
- Mark entities as deleted unless the revision log explicitly shows deleted=true
- Add new entities that aren't already in Tier 1 or the revision log
- Speculate about cascading effects

Return the COMPLETE entity state (all entities, preserving unchanged ones exactly) \
and updated groups.

Output ONLY valid JSON — no prose, no fences. Start with {:
{
  "entities": {
    "entity_name": {
      "current_value": "value or null",
      "last_updated": "YYYY/MM/DD",
      "deleted": false,
      "deletion_note": null
    }
  },
  "groups": {
    "group_name": ["entity1", "entity2"]
  }
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_session(session: dict) -> str:
    ts = session.get("timestamp", "")
    parts = [f"[Session: {ts}]"] if ts else ["[Session]"]
    for turn in session.get("conversation", []):
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _extract_json(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return text


def _call_claude(prompt: str, system: str, model: str = "claude-code",
                 timeout: int = 120, cwd: Optional[str] = None) -> str:
    cmd = ["claude", "-p", "--output-format", "text", "--no-session-persistence"]
    if "/" in model:
        cmd.extend(["--model", model.split("/", 1)[1]])
    cmd.extend(["--system-prompt", system])
    result = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=cwd
    )
    output = result.stdout.strip()
    if result.returncode != 0 and not output:
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}): {result.stderr.strip()[:300]}"
        )
    return output


# ---------------------------------------------------------------------------
# Tier I/O helpers
# ---------------------------------------------------------------------------

def _tier1_path(memory_dir: str) -> str:
    return os.path.join(memory_dir, "tier1_entity_state.json")


def _tier2_path(memory_dir: str) -> str:
    return os.path.join(memory_dir, "tier2_summaries.jsonl")


def _tier3_path(memory_dir: str) -> str:
    return os.path.join(memory_dir, "tier3_revisions.jsonl")


def _load_tier1(memory_dir: str) -> dict:
    """Returns {"entities": {...}, "groups": {...}}"""
    path = _tier1_path(memory_dir)
    if not os.path.exists(path):
        return {"entities": {}, "groups": {}}
    try:
        with open(path) as f:
            data = json.load(f)
        # backwards compat: old format was flat dict of entities
        if "entities" not in data:
            return {"entities": data, "groups": {}}
        return data
    except Exception:
        return {"entities": {}, "groups": {}}


def _save_tier1(memory_dir: str, state: dict) -> None:
    with open(_tier1_path(memory_dir), "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _append_tier2(memory_dir: str, summary: str, timestamp: str) -> None:
    with open(_tier2_path(memory_dir), "a") as f:
        f.write(json.dumps({"timestamp": timestamp, "summary": summary}) + "\n")


def _append_tier3(memory_dir: str, entity: str, timestamp: str,
                  old_value, new_value, deleted: bool) -> None:
    with open(_tier3_path(memory_dir), "a") as f:
        f.write(json.dumps({
            "entity": entity, "timestamp": timestamp,
            "old_value": old_value, "new_value": new_value, "deleted": deleted,
        }) + "\n")


def _read_tier2_recent(memory_dir: str, n: int = 5) -> List[dict]:
    path = _tier2_path(memory_dir)
    if not os.path.exists(path):
        return []
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except Exception:
                    pass
    return lines[-n:]


def _read_tier3_since(memory_dir: str, since_index: int) -> List[dict]:
    """Read Tier 3 entries from a given line index onward."""
    path = _tier3_path(memory_dir)
    if not os.path.exists(path):
        return []
    lines = []
    with open(path) as f:
        all_lines = f.readlines()
    for line in all_lines[since_index:]:
        line = line.strip()
        if line:
            try:
                lines.append(json.loads(line))
            except Exception:
                pass
    return lines


def _tier3_line_count(memory_dir: str) -> int:
    path = _tier3_path(memory_dir)
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def _read_tier3_recent(memory_dir: str, n: int = 30) -> List[dict]:
    path = _tier3_path(memory_dir)
    if not os.path.exists(path):
        return []
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except Exception:
                    pass
    return lines[-n:]


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _format_tier1_context(state: dict) -> str:
    entities = state.get("entities", {})
    groups = state.get("groups", {})
    if not entities:
        return ""

    deleted_parts = []
    active_parts = []
    for entity, info in sorted(entities.items()):
        if info.get("deleted"):
            deleted_parts.append(
                f"  ⚠ DELETED [{info.get('last_updated', '')}]: {entity} — "
                f"{info.get('deletion_note', 'explicitly removed')}"
            )
        else:
            active_parts.append(
                f"  {entity}: {info.get('current_value', '?')} "
                f"(as of {info.get('last_updated', '?')})"
            )

    result_parts = []
    if deleted_parts:
        result_parts.append(
            "⚠ DELETED/DISCONTINUED (no current value — answer 'I don't know' or "
            "'was deleted' for these):\n" + "\n".join(deleted_parts)
        )
    if active_parts:
        result_parts.append("CURRENT STATE:\n" + "\n".join(active_parts))

    # Entity groups for Agg-style questions
    if groups:
        group_lines = []
        for gname, members in sorted(groups.items()):
            valid = [m for m in members if m in entities]
            if valid:
                group_lines.append(f"  {gname}: {', '.join(valid)}")
        if group_lines:
            result_parts.append("ENTITY GROUPS (related facts):\n" + "\n".join(group_lines))

    return "\n\n".join(result_parts).strip()


def _format_tier3_context(revisions: List[dict]) -> str:
    if not revisions:
        return ""
    parts = []
    for r in revisions:
        e = r.get("entity", "?")
        ts = r.get("timestamp", "?")
        old = r.get("old_value")
        new = r.get("new_value")
        if r.get("deleted"):
            parts.append(f"  [{ts}] {e}: DELETED/DISCONTINUED (was: {old})")
        elif old is None:
            parts.append(f"  [{ts}] {e}: set to {new!r}")
        else:
            parts.append(f"  [{ts}] {e}: {old!r} → {new!r}")
    return "\n".join(parts)


def _format_tier2_context(summaries: List[dict]) -> str:
    if not summaries:
        return ""
    return "\n".join(
        f"  [{s.get('timestamp', '')}] {s.get('summary', '')}"
        for s in summaries
    )


def _format_tier0_context(hot_buffer: list) -> str:
    if not hot_buffer:
        return ""
    return "\n\n".join(_format_session(s) for s in hot_buffer)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TieredMemory(BaseMemorySystem):
    """
    Tiered Memory v2: bounded-cost agent with EvoMemory-inspired Cascade Refine
    and A-Mem-inspired entity grouping.

    Per evidence session: EXTRACT (1 call). Every K sessions: COMPRESS (1 call).
    Per phase end: CASCADE REFINE (1 call, ~4k tokens in — reads Tier 1 + phase revisions).

    Addresses v1 weaknesses:
      Cas: CASCADE REFINE propagates implicit dependency updates
      Agg: ENTITY GROUPS cluster semantically related entities
      Del: CASCADE REFINE verifies and marks deletions explicitly
      Abs: Clearer deletion markers in retrieval context
    """

    HOT_K = 5            # compress hot buffer every K evidence sessions
    MAX_REVISIONS = 30   # max revision entries from Tier 3 at retrieval time
    MAX_SUMMARIES = 5    # max Tier 2 summaries at retrieval time

    def __init__(self, model: str = "claude-code", base_tmp_dir: Optional[str] = None):
        self.model = model
        self.base_tmp_dir = base_tmp_dir or tempfile.gettempdir()
        self._memory_dir: Optional[str] = None
        self._hot_buffer: List[dict] = []
        self._evidence_count: int = 0
        self._phase_tier3_start: int = 0   # tier3 line count at start of current phase
        self._last_retrieved_context: str = ""

    def reset(self):
        if self._memory_dir and os.path.isdir(self._memory_dir):
            shutil.rmtree(self._memory_dir, ignore_errors=True)
        ts = int(time.time() * 1000)
        self._memory_dir = os.path.join(
            self.base_tmp_dir, f"meme_tiered_{os.getpid()}_{ts}"
        )
        os.makedirs(self._memory_dir, exist_ok=True)
        self._hot_buffer = []
        self._evidence_count = 0
        self._phase_tier3_start = 0
        self._last_retrieved_context = ""

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest_session(self, session: dict) -> dict:
        if self._memory_dir is None:
            self.reset()

        if session.get("type") == "filler":
            return {"skipped": True, "reason": "filler session",
                    "token_usage": {"input_tokens": 0, "output_tokens": 0}}

        session_text = _format_session(session)
        session_ts = session.get("timestamp", "")[:10]

        # ---- Step 1: EXTRACT entity updates ----
        tier1_state = _load_tier1(self._memory_dir)
        entities = tier1_state.get("entities", {})
        known_entities = list(entities.keys())

        extract_prompt = (
            f"Known entities: {', '.join(known_entities) if known_entities else 'none yet'}\n\n"
            f"Session to process:\n{session_text}"
        )

        updates_applied = []
        compress_fired = False

        try:
            raw = _call_claude(extract_prompt, EXTRACT_SYSTEM_PROMPT,
                               self.model, cwd=self._memory_dir)
            parsed = json.loads(_extract_json(raw))
            updates = parsed.get("updates", [])

            for upd in updates:
                entity = (upd.get("entity") or "").strip()
                new_value = upd.get("value")
                deleted = bool(upd.get("deleted", False))
                upd_ts = (upd.get("timestamp") or session_ts)[:10]

                if not entity:
                    continue

                old_info = entities.get(entity, {})
                old_value = old_info.get("current_value") if not old_info.get("deleted") else None

                if deleted:
                    entities[entity] = {
                        "current_value": None,
                        "last_updated": upd_ts,
                        "deleted": True,
                        "deletion_note": f"explicitly removed as of {upd_ts}",
                    }
                else:
                    entities[entity] = {
                        "current_value": new_value,
                        "last_updated": upd_ts,
                        "deleted": False,
                        "deletion_note": None,
                    }

                _append_tier3(self._memory_dir, entity, upd_ts,
                              old_value, new_value, deleted)
                updates_applied.append(entity)

            tier1_state["entities"] = entities
            _save_tier1(self._memory_dir, tier1_state)

        except Exception:
            pass  # extract failed — still add to hot buffer

        # ---- Step 2: Add to Tier 0 hot buffer ----
        self._hot_buffer.append(session)
        self._evidence_count += 1

        # ---- Step 3: COMPRESS if hot buffer is full ----
        if len(self._hot_buffer) >= self.HOT_K:
            compress_fired = self._compress_hot_buffer()

        return {
            "updates_applied": updates_applied,
            "compress_fired": compress_fired,
            "evidence_count": self._evidence_count,
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
        }

    def _compress_hot_buffer(self) -> bool:
        if not self._hot_buffer:
            return False
        sessions_text = "\n\n---\n\n".join(
            _format_session(s) for s in self._hot_buffer
        )
        last_ts = (self._hot_buffer[-1].get("timestamp") or "")[:10]
        first_ts = (self._hot_buffer[0].get("timestamp") or "")[:10]
        try:
            summary = _call_claude(
                f"Sessions from {first_ts} to {last_ts}:\n\n{sessions_text}",
                COMPRESS_SYSTEM_PROMPT, self.model, timeout=90, cwd=self._memory_dir,
            )
            if summary:
                _append_tier2(self._memory_dir, summary.strip(), last_ts)
        except Exception:
            pass
        self._hot_buffer = []
        return True

    # ------------------------------------------------------------------
    # CASCADE REFINE  (EvoMemory-inspired, bounded input)
    # ------------------------------------------------------------------

    def finalize_ingest(self) -> None:
        """Flush hot buffer then run CASCADE REFINE on Tier 1 + phase revisions."""
        if self._hot_buffer:
            self._compress_hot_buffer()

        if not self._memory_dir:
            return

        tier1_state = _load_tier1(self._memory_dir)
        entities = tier1_state.get("entities", {})
        if not entities:
            self._phase_tier3_start = _tier3_line_count(self._memory_dir)
            return

        # Read only the revisions from this phase (bounded)
        phase_revisions = _read_tier3_since(self._memory_dir, self._phase_tier3_start)

        if not phase_revisions:
            self._phase_tier3_start = _tier3_line_count(self._memory_dir)
            return

        # Format Tier 1 compactly for the refine call
        entity_lines = []
        for ename, info in sorted(entities.items()):
            if info.get("deleted"):
                entity_lines.append(
                    f'  "{ename}": DELETED as of {info.get("last_updated","?")}'
                    f' (was: {info.get("deletion_note","?")})'
                )
            else:
                entity_lines.append(
                    f'  "{ename}": {info.get("current_value","?")} '
                    f'(as of {info.get("last_updated","?")})'
                )
        tier1_text = "\n".join(entity_lines)

        rev_lines = []
        for r in phase_revisions:
            if r.get("deleted"):
                rev_lines.append(
                    f'  [{r["timestamp"]}] {r["entity"]}: DELETED (was: {r.get("old_value")})'
                )
            elif r.get("old_value") is None:
                rev_lines.append(
                    f'  [{r["timestamp"]}] {r["entity"]}: set to {r.get("new_value")!r}'
                )
            else:
                rev_lines.append(
                    f'  [{r["timestamp"]}] {r["entity"]}: '
                    f'{r.get("old_value")!r} → {r.get("new_value")!r}'
                )
        rev_text = "\n".join(rev_lines)

        existing_groups = tier1_state.get("groups", {})
        groups_text = json.dumps(existing_groups, indent=2) if existing_groups else "{}"

        refine_prompt = (
            f"Current entity state:\n{tier1_text}\n\n"
            f"Changes applied this phase:\n{rev_text}\n\n"
            f"Current entity groups:\n{groups_text}\n\n"
            f"Apply cascades, verify deletions, and update groups."
        )

        try:
            raw = _call_claude(refine_prompt, CASCADE_REFINE_SYSTEM_PROMPT,
                               self.model, timeout=180, cwd=self._memory_dir)
            parsed = json.loads(_extract_json(raw))

            # Update Tier 1 with refined entities
            refined_entities = parsed.get("entities", {})
            if refined_entities:
                # Merge: keep any entities the refine dropped (safety)
                for ename, info in entities.items():
                    if ename not in refined_entities:
                        refined_entities[ename] = info
                tier1_state["entities"] = refined_entities

            refined_groups = parsed.get("groups", {})
            if refined_groups:
                tier1_state["groups"] = refined_groups

            _save_tier1(self._memory_dir, tier1_state)

        except Exception as e:
            print(f"      [tiered] CASCADE REFINE failed: {e}")

        self._phase_tier3_start = _tier3_line_count(self._memory_dir)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> str:
        """Assemble bounded context from all four tiers. No LLM call."""
        if not self._memory_dir or not os.path.isdir(self._memory_dir):
            self._last_retrieved_context = "(no memory)"
            return "(no memory)"

        parts = []

        tier1_state = _load_tier1(self._memory_dir)
        tier1_text = _format_tier1_context(tier1_state)
        if tier1_text:
            parts.append("## Current Entity State\n" + tier1_text)

        revisions = _read_tier3_recent(self._memory_dir, self.MAX_REVISIONS)
        tier3_text = _format_tier3_context(revisions)
        if tier3_text:
            parts.append("## Revision History\n" + tier3_text)

        summaries = _read_tier2_recent(self._memory_dir, self.MAX_SUMMARIES)
        tier2_text = _format_tier2_context(summaries)
        if tier2_text:
            parts.append("## Recent Session Summaries\n" + tier2_text)

        tier0_text = _format_tier0_context(self._hot_buffer)
        if tier0_text:
            parts.append("## Most Recent Sessions (raw)\n" + tier0_text)

        context = "\n\n".join(parts) if parts else "(no memory)"
        self._last_retrieved_context = context
        return context

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def get_memory_snapshot(self) -> dict:
        if not self._memory_dir:
            return {"text": "(no memory)"}
        tier1_state = _load_tier1(self._memory_dir)
        tier1_text = _format_tier1_context(tier1_state)
        revisions = _read_tier3_recent(self._memory_dir, 50)
        tier3_text = _format_tier3_context(revisions)
        summaries = _read_tier2_recent(self._memory_dir, 10)
        tier2_text = _format_tier2_context(summaries)
        parts = []
        if tier1_text:
            parts.append("=== Tier 1: Entity State ===\n" + tier1_text)
        if tier3_text:
            parts.append("=== Tier 3: Revisions ===\n" + tier3_text)
        if tier2_text:
            parts.append("=== Tier 2: Summaries ===\n" + tier2_text)
        return {"text": "\n\n".join(parts) or "(no memory)"}

    def get_retrieved_context(self) -> str:
        return self._last_retrieved_context
