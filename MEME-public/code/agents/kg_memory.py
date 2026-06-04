"""
Knowledge Graph Memory agent for MeME evaluation.

Obsidian-style wiki files as a temporal knowledge graph.

Storage layout (per episode temp dir):
  pages/{entity}.md   — one file per entity (graph node)
                         YAML frontmatter: current_value, timestamps, deleted, tombstone
                         Relationships: typed [[wiki-links]] (graph edges)
                         History table: append-only value changelog
  INDEX.md            — quick-lookup: active entities + deleted tombstones
  summaries.jsonl     — compressed session context (for Agg fallback)

Ingest (per evidence session, 1 LLM call):
  EXTRACT+WRITE — reads session + current entity files → outputs updated .md files
                  with typed [[links]] and appended history rows

Post-phase (finalize_ingest, 1 LLM call if changes exist):
  CASCADE — grep reverse [[links]] for each changed entity → batch LLM call:
            "did any of these linked entities also change in this session?"

Retrieve (no LLM call):
  Load INDEX.md → load relevant entity pages → follow [[links]] 1 hop →
  assemble grouped context: DELETED at top, active entities with relationships,
  history section for Tr queries.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Dict, List, Optional, Set

from agents.base import BaseMemorySystem


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACT_WRITE_SYSTEM = """\
You are maintaining a knowledge graph memory. Each entity is stored as a markdown file.

FILE FORMAT (follow exactly):
---
entity: snake_case_name
current_value: "exact verbatim value"
first_seen: "YYYY/MM/DD"
last_updated: "YYYY/MM/DD"
deleted: false
tombstone: null
---

# entity_name

**Current**: exact verbatim value

## Relationships
- treats → [[health_condition]]
- proximate_to → [[workplace]]

## History
| Date | Value |
|------|-------|
| YYYY/MM/DD | initial value |

RULES:
- Entity names: snake_case, match existing names exactly
- current_value: store VERBATIM — do not paraphrase or summarize
- Relationships: typed directed edges (treats, affects, proximate_to, destination,
  requires, implies, part_of, managed_by, lives_with, related_to)
- History: APPEND new rows only, NEVER remove existing rows
- Deletion: set deleted=true, tombstone="Was: X. Reason. No replacement.", body starts with DELETED
- Only output files for entities that actually changed
- If nothing changed, output: {"files": []}

Output ONLY valid JSON starting with {
"""

CASCADE_SYSTEM = """\
Some entities changed. Check if any LINKED entities also changed in the session.
Only update a linked entity if the session EXPLICITLY states its value changed.
Do NOT speculate or infer cascades.
Output ONLY valid JSON starting with {
If nothing changed: {"files": []}
"""

COMPRESS_PROMPT = """\
Write 2-3 sentences summarizing the key facts from these conversation sessions.
Focus on what was established or changed. Be specific: names, values, dates.
Output ONLY plain text — no JSON, no headers.
"""


# ---------------------------------------------------------------------------
# Markdown / graph helpers
# ---------------------------------------------------------------------------

def _parse_frontmatter(content: str) -> dict:
    """Extract YAML-ish frontmatter fields from markdown."""
    fm = {}
    if not content.startswith("---"):
        return fm
    end = content.find("---", 3)
    if end == -1:
        return fm
    block = content[3:end]
    for line in block.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if v.lower() == "true":
                v = True
            elif v.lower() == "false":
                v = False
            elif v.lower() == "null":
                v = None
            fm[k] = v
    return fm


def _parse_links(content: str) -> List[str]:
    """Extract [[entity_name]] link targets from markdown content."""
    return re.findall(r'\[\[([^\]\|]+)\]\]', content)


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
                 timeout: int = 180, cwd: Optional[str] = None) -> str:
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
# Pages directory helpers
# ---------------------------------------------------------------------------

def _pages_dir(memory_dir: str) -> str:
    d = os.path.join(memory_dir, "pages")
    os.makedirs(d, exist_ok=True)
    return d


def _page_path(memory_dir: str, entity: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", entity)
    return os.path.join(_pages_dir(memory_dir), f"{safe}.md")


def _load_page(memory_dir: str, entity: str) -> Optional[str]:
    path = _page_path(memory_dir, entity)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read()


def _save_page(memory_dir: str, entity: str, content: str) -> None:
    with open(_page_path(memory_dir, entity), "w") as f:
        f.write(content.rstrip() + "\n")


def _all_entities(memory_dir: str) -> List[str]:
    pages = _pages_dir(memory_dir)
    return [f[:-3] for f in os.listdir(pages) if f.endswith(".md")]


def _load_all_pages(memory_dir: str) -> Dict[str, str]:
    pages = _pages_dir(memory_dir)
    result = {}
    for fname in os.listdir(pages):
        if not fname.endswith(".md"):
            continue
        entity = fname[:-3]
        with open(os.path.join(pages, fname)) as f:
            result[entity] = f.read()
    return result


def _reverse_links(changed_entity: str, memory_dir: str) -> List[str]:
    """Find all entities whose pages contain [[changed_entity]]."""
    pattern = f"[[{changed_entity}]]"
    results = []
    pages = _pages_dir(memory_dir)
    for fname in os.listdir(pages):
        if not fname.endswith(".md"):
            continue
        entity = fname[:-3]
        if entity == changed_entity:
            continue
        try:
            with open(os.path.join(pages, fname)) as f:
                if pattern in f.read():
                    results.append(entity)
        except OSError:
            pass
    return results


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def _rebuild_index(memory_dir: str) -> None:
    all_pages = _load_all_pages(memory_dir)
    active, deleted = [], []
    for entity, content in sorted(all_pages.items()):
        fm = _parse_frontmatter(content)
        val = fm.get("current_value") or "?"
        ts = fm.get("last_updated") or "?"
        if fm.get("deleted"):
            tomb = fm.get("tombstone") or f"Was: {val}"
            deleted.append(f"- [[{entity}]] — ⚠ DELETED ({ts}): {tomb}")
        else:
            active.append(f"- [[{entity}]] — {val} ({ts})")

    lines = ["# Entity Index\n"]
    if deleted:
        lines.append("## ⚠ Deleted (no current value)\n")
        lines.extend(deleted)
        lines.append("")
    if active:
        lines.append("## Active\n")
        lines.extend(active)

    with open(os.path.join(memory_dir, "INDEX.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _read_index(memory_dir: str) -> str:
    path = os.path.join(memory_dir, "INDEX.md")
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Summaries helpers
# ---------------------------------------------------------------------------

def _summaries_path(memory_dir: str) -> str:
    return os.path.join(memory_dir, "summaries.jsonl")


def _append_summary(memory_dir: str, summary: str, timestamp: str) -> None:
    with open(_summaries_path(memory_dir), "a") as f:
        f.write(json.dumps({"ts": timestamp, "summary": summary}) + "\n")


def _read_summaries(memory_dir: str, n: int = 5) -> List[dict]:
    path = _summaries_path(memory_dir)
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
# Apply LLM file output
# ---------------------------------------------------------------------------

def _apply_file_output(memory_dir: str, raw_output: str) -> List[str]:
    """Parse JSON file list and write pages. Returns list of entity names written."""
    written = []
    try:
        parsed = json.loads(_extract_json(raw_output))
        for fspec in parsed.get("files", []):
            name = (fspec.get("name") or fspec.get("path") or "").strip()
            content = (fspec.get("content") or "").strip()
            if not name or not content:
                continue
            # Normalise to snake_case entity name
            entity = re.sub(r"\.md$", "", name)
            entity = re.sub(r"[^\w\-]", "_", entity.strip())
            _save_page(memory_dir, entity, content)
            written.append(entity)
    except Exception:
        pass
    return written


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _assemble_context(memory_dir: str, question: str) -> str:
    all_pages = _load_all_pages(memory_dir)
    if not all_pages:
        return "(no memory)"

    # Separate deleted from active
    active: Dict[str, dict] = {}   # entity → {fm, content, links}
    deleted: Dict[str, dict] = {}

    for entity, content in all_pages.items():
        fm = _parse_frontmatter(content)
        links = _parse_links(content)
        entry = {"fm": fm, "content": content, "links": links}
        if fm.get("deleted"):
            deleted[entity] = entry
        else:
            active[entity] = entry

    parts = []

    # 1. Deleted entities — prominent, with tombstones
    if deleted:
        del_lines = ["## ⚠ DELETED — No Current Value",
                     "(For these entities: answer 'was deleted' or 'I don't know')\n"]
        for entity, e in sorted(deleted.items()):
            fm = e["fm"]
            tomb = fm.get("tombstone") or f"Was: {fm.get('current_value', '?')}"
            del_lines.append(f"- **{entity}** [{fm.get('last_updated','?')}]: {tomb}")
        parts.append("\n".join(del_lines))

    # 2. Active entity pages — full content for rich context
    if active:
        entity_parts = []
        for entity, e in sorted(active.items()):
            # Include full page content so LLM sees relationships + history
            entity_parts.append(e["content"].strip())
        parts.append("## Entity Pages\n\n" + "\n\n---\n\n".join(entity_parts))

    # 3. Recent session summaries
    summaries = _read_summaries(memory_dir, 3)
    if summaries:
        sum_lines = ["## Recent Context"]
        for s in summaries:
            sum_lines.append(f"  [{s.get('ts','')}] {s.get('summary','')}")
        parts.append("\n".join(sum_lines))

    return "\n\n".join(parts) if parts else "(no memory)"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class KGMemory(BaseMemorySystem):
    """
    Knowledge Graph Memory: Obsidian wiki files as a temporal knowledge graph.

    Each entity is a markdown file with YAML frontmatter (verbatim current value,
    deleted flag, timestamps) + typed [[wiki-link]] edges + append-only history table.

    Cascade detection uses reverse [[link]] grep — no graph database needed.
    Retrieval assembles context from all entity pages + linked neighbors.
    """

    HOT_K = 5   # compress hot buffer every K evidence sessions

    def __init__(self, model: str = "claude-code", base_tmp_dir: Optional[str] = None):
        self.model = model
        self.base_tmp_dir = base_tmp_dir or tempfile.gettempdir()
        self._memory_dir: Optional[str] = None
        self._hot_buffer: List[dict] = []
        self._phase_evidence: List[dict] = []   # evidence sessions this phase
        self._phase_changed: Set[str] = set()   # entities changed this phase
        self._last_retrieved_context: str = ""

    def reset(self):
        if self._memory_dir and os.path.isdir(self._memory_dir):
            shutil.rmtree(self._memory_dir, ignore_errors=True)
        ts = int(time.time() * 1000)
        self._memory_dir = os.path.join(
            self.base_tmp_dir, f"meme_kg_{os.getpid()}_{ts}"
        )
        os.makedirs(self._memory_dir, exist_ok=True)
        _pages_dir(self._memory_dir)
        self._hot_buffer = []
        self._phase_evidence = []
        self._phase_changed = set()
        self._last_retrieved_context = ""

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest_session(self, session: dict) -> dict:
        if self._memory_dir is None:
            self.reset()

        if session.get("type") == "filler":
            self._hot_buffer.append(session)
            if len(self._hot_buffer) >= self.HOT_K:
                self._compress_hot_buffer()
            return {"skipped": True, "reason": "filler session",
                    "token_usage": {"input_tokens": 0, "output_tokens": 0}}

        session_text = _format_session(session)
        self._phase_evidence.append(session)

        # Build current files context for the prompt
        all_pages = _load_all_pages(self._memory_dir)
        if all_pages:
            current_files = "\n\n---\n\n".join(
                f"### {entity}.md\n{content.strip()}"
                for entity, content in sorted(all_pages.items())
            )
        else:
            current_files = "(none yet)"

        prompt = (
            f"## Current entity files\n\n{current_files}\n\n"
            f"## Session to process\n\n{session_text}"
        )

        written = []
        try:
            raw = _call_claude(prompt, EXTRACT_WRITE_SYSTEM,
                               self.model, timeout=180, cwd=self._memory_dir)
            written = _apply_file_output(self._memory_dir, raw)
            self._phase_changed.update(written)
            if written:
                _rebuild_index(self._memory_dir)
        except Exception as e:
            pass  # session skipped on error

        # Hot buffer for compression (filler + evidence mixed)
        self._hot_buffer.append(session)
        if len(self._hot_buffer) >= self.HOT_K:
            self._compress_hot_buffer()

        return {
            "pages_written": written,
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
        }

    def _compress_hot_buffer(self) -> None:
        if not self._hot_buffer:
            return
        sessions_text = "\n\n---\n\n".join(_format_session(s) for s in self._hot_buffer)
        last_ts = (self._hot_buffer[-1].get("timestamp") or "")[:10]
        first_ts = (self._hot_buffer[0].get("timestamp") or "")[:10]
        try:
            summary = _call_claude(
                f"Sessions from {first_ts} to {last_ts}:\n\n{sessions_text}",
                "Write 2-3 sentences summarizing the key facts. Be specific: names, values, dates. Output ONLY plain text.",
                self.model, timeout=60, cwd=self._memory_dir,
            )
            if summary:
                _append_summary(self._memory_dir, summary.strip(), last_ts)
        except Exception:
            pass
        self._hot_buffer = []

    # ------------------------------------------------------------------
    # CASCADE  (post-phase, EvoMemory-inspired but graph-targeted)
    # ------------------------------------------------------------------

    def finalize_ingest(self) -> None:
        """Flush hot buffer, then cascade changed entities to linked neighbors."""
        if self._hot_buffer:
            self._compress_hot_buffer()

        if not self._memory_dir or not self._phase_changed:
            self._phase_evidence = []
            self._phase_changed = set()
            return

        # Find all entities with reverse [[links]] to any changed entity
        all_affected: Set[str] = set()
        for entity in self._phase_changed:
            for affected in _reverse_links(entity, self._memory_dir):
                if affected not in self._phase_changed:
                    all_affected.add(affected)

        if all_affected:
            # Build changed summary
            changed_lines = []
            for entity in sorted(self._phase_changed):
                content = _load_page(self._memory_dir, entity) or ""
                fm = _parse_frontmatter(content)
                if fm.get("deleted"):
                    changed_lines.append(
                        f"- {entity}: DELETED (was: {fm.get('tombstone', '?')})"
                    )
                else:
                    changed_lines.append(
                        f"- {entity}: {fm.get('current_value', '?')}"
                    )

            # Build linked files context
            linked_parts = []
            for entity in sorted(all_affected):
                content = _load_page(self._memory_dir, entity)
                if content:
                    linked_parts.append(f"### {entity}.md\n{content.strip()}")

            # Use the most recent evidence session text
            session_text = "\n\n---\n\n".join(
                _format_session(s) for s in self._phase_evidence[-2:]
            ) if self._phase_evidence else "(no session text)"

            linked_text = "\n\n".join(linked_parts) if linked_parts else "(none)"
            prompt = (
                f"## Entities that changed this phase\n\n{chr(10).join(changed_lines)}\n\n"
                f"## Linked entities to check\n\n{linked_text}\n\n"
                f"## Evidence session text\n\n{session_text[:4000]}"
            )

            try:
                raw = _call_claude(
                    prompt, CASCADE_SYSTEM,
                    self.model, timeout=180, cwd=self._memory_dir,
                )
                cascade_written = _apply_file_output(self._memory_dir, raw)
                if cascade_written:
                    _rebuild_index(self._memory_dir)
                    print(f"      [kg] cascade: {cascade_written}")
            except Exception as e:
                print(f"      [kg] cascade failed: {e}")

        self._phase_evidence = []
        self._phase_changed = set()

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> str:
        """Return full graph context — no LLM call."""
        if not self._memory_dir or not os.path.isdir(self._memory_dir):
            self._last_retrieved_context = "(no memory)"
            return "(no memory)"

        context = _assemble_context(self._memory_dir, question)
        self._last_retrieved_context = context
        return context

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def get_memory_snapshot(self) -> dict:
        if not self._memory_dir:
            return {"text": "(no memory)"}
        index = _read_index(self._memory_dir)
        all_pages = _load_all_pages(self._memory_dir)
        pages_text = "\n\n---\n\n".join(
            f"# {e}\n{c.strip()}" for e, c in sorted(all_pages.items())
        )
        return {"text": f"{index}\n\n{pages_text}" if pages_text else "(no memory)"}

    def get_retrieved_context(self) -> str:
        return self._last_retrieved_context
