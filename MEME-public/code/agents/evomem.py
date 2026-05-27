"""
EvoMemory agent for MeME evaluation.

Inspired by ReMem (Wei et al., 2026 — Evo-Memory paper): memory is not just
passively written and read, it is actively *refined* — pruned, consolidated,
and made coherent — after each ingest phase.

Architecture:
  ingest_session():   Same as auto_memory — claude -p writes/updates typed .md
                      files for each evidence session. Filler sessions skipped.
  finalize_ingest():  REFINE step (the EvoMemory contribution) — a single
                      claude -p call reads all current memory files and:
                        • Merges duplicate / overlapping facts
                        • Resolves contradictions (most-recent wins)
                        • Ensures deletions are clearly marked at the top
                        • Removes noise and irrelevant entries
                        • Consolidates same-entity facts into one file
  retrieve():         Direct file read, no LLM call — same as auto_memory.
                      With Refine having cleaned the files, they are always
                      coherent going into question time.

vs. auto_memory:  adds 1 claude -p Refine call per phase (at finalize_ingest)
vs. wiki:         no per-question LLM call; Refine replaces index-guided nav
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

INGEST_SYSTEM_PROMPT = """\
You are a memory manager. Read a new conversation session and decide what \
information is worth saving to persistent memory files.

## Memory file format

Each file uses YAML frontmatter followed by body content:

---
name: short-kebab-slug
description: one-line summary
metadata:
  type: user | feedback | project | reference
---

Body content. Be concise — one fact per line when possible.

## Memory types
- user: facts about who the user is (role, goals, expertise, preferences)
- feedback: how to work with them (what to avoid, confirmed good approaches)
- project: ongoing work context (current tasks, decisions, goals, key facts)
- reference: pointers to external systems (where things are tracked, tools used)

## Rules
- Save only information useful in FUTURE sessions — skip pleasantries and filler
- Update existing files instead of creating duplicates
- If nothing useful in the session, return empty files list
- Keep file names short (snake_case.md)

## Output format
Output ONLY valid JSON — no prose, no markdown fences. Start with {:
{"files": [{"name": "filename.md", "content": "---\\nname: ...\\n...\\n---\\n\\nbody"}], "delete": []}
"""

REFINE_SYSTEM_PROMPT = """\
You are refining a memory store for coherence and accuracy. This is the \
EvoMemory Refine step: actively reorganize memory so it is consistent and \
ready for retrieval.

You will receive all current memory files. Produce a revised set that is:

1. CONSOLIDATED — facts about the same entity/topic in one file, not scattered
2. CONTRADICTION-FREE — when facts conflict, keep the most recent; mark old \
   values as "previously: X" if useful
3. DELETION-AWARE — facts the user explicitly removed/cancelled must be clearly \
   marked at the TOP of the file as "DELETED/DISCONTINUED: <fact>" so retrieval \
   never returns them as current
4. NOISE-FREE — remove greetings, filler, redundant restatements, and any fact \
   that has been superseded
5. COMPLETE — every surviving fact should have enough context to be understood \
   in isolation

## Output format
Output ONLY valid JSON — no prose, no markdown fences. Start with {:
{"files": [{"name": "filename.md", "content": "---\\nname: ...\\n...\\n---\\n\\nbody"}], "delete": ["obsolete.md"]}

If the memory is already clean and coherent, return the files unchanged.
If there is no memory to refine, return: {"files": [], "delete": []}
"""


# ---------------------------------------------------------------------------
# Shared helpers (identical to auto_memory helpers)
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


def _read_memory_files(memory_dir: str) -> str:
    """Concatenate all .md files except MEMORY.md with headers."""
    if not os.path.isdir(memory_dir):
        return ""
    parts = []
    for fname in sorted(os.listdir(memory_dir)):
        if fname == "MEMORY.md" or not fname.endswith(".md"):
            continue
        fpath = os.path.join(memory_dir, fname)
        try:
            with open(fpath) as f:
                content = f.read().strip()
            if content:
                parts.append(f"### {fname}\n{content}")
        except OSError:
            pass
    return "\n\n".join(parts)


def _strip_frontmatter(content: str) -> str:
    content = content.strip()
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            return content[end + 3:].strip()
    return content


def _rebuild_memory_index(memory_dir: str) -> None:
    entries = []
    for fname in sorted(os.listdir(memory_dir)):
        if fname == "MEMORY.md" or not fname.endswith(".md"):
            continue
        fpath = os.path.join(memory_dir, fname)
        desc = fname
        try:
            with open(fpath) as f:
                text = f.read()
            m = re.search(r"description:\s*(.+)", text)
            if m:
                desc = m.group(1).strip()
        except OSError:
            pass
        entries.append(f"- [{fname}]({fname}) — {desc}")

    index_path = os.path.join(memory_dir, "MEMORY.md")
    with open(index_path, "w") as f:
        if entries:
            f.write("# Memory Index\n\n" + "\n".join(entries) + "\n")
        else:
            f.write("# Memory Index\n\n(empty)\n")


def _extract_json(text: str) -> str:
    """Extract JSON from text that may contain prose or code fences."""
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
        sub_model = model.split("/", 1)[1]
        cmd.extend(["--model", sub_model])
    cmd.extend(["--system-prompt", system])
    result = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, timeout=timeout,
        cwd=cwd,
    )
    output = result.stdout.strip()
    if result.returncode != 0 and not output:
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}): {result.stderr.strip()[:300]}"
        )
    return output


def _apply_file_changes(memory_dir: str, raw_output: str,
                        fallback_id: str = "unknown") -> tuple:
    """Parse JSON output and write/delete memory files. Returns (written, deleted)."""
    files_written: List[str] = []
    files_deleted: List[str] = []
    text = _extract_json(raw_output)

    try:
        parsed = json.loads(text)

        for fspec in parsed.get("files", []):
            fname = re.sub(r"[^\w\-.]", "_", (fspec.get("name") or "").strip())
            content = (fspec.get("content") or "").strip()
            if not fname or not content:
                continue
            if not fname.endswith(".md"):
                fname += ".md"
            with open(os.path.join(memory_dir, fname), "w") as f:
                f.write(content + "\n")
            files_written.append(fname)

        for fname in parsed.get("delete", []):
            fname = re.sub(r"[^\w\-.]", "_", (fname or "").strip())
            fpath = os.path.join(memory_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
                files_deleted.append(fname)

    except (json.JSONDecodeError, KeyError):
        # Non-JSON fallback: save raw output as a note
        if text.strip():
            safe_id = re.sub(r"\W", "_", fallback_id)[:20]
            fname = f"notes_{safe_id}.md"
            with open(os.path.join(memory_dir, fname), "w") as f:
                slug = fname[:-3]
                f.write(
                    f"---\nname: {slug}\ndescription: session notes\n"
                    f"metadata:\n  type: project\n---\n\n{text.strip()}\n"
                )
            files_written.append(fname)

    return files_written, files_deleted


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class EvoMemory(BaseMemorySystem):
    """
    EvoMemory agent: auto-memory ingest + ReMem-inspired Refine phase.

    The Refine step (finalize_ingest) is what distinguishes this from
    auto_memory: after all sessions in a phase are ingested, one additional
    claude -p call reorganizes the entire memory store for coherence —
    merging duplicates, resolving contradictions, and surfacing deletions.
    """

    def __init__(self, model: str = "claude-code",
                 base_tmp_dir: Optional[str] = None):
        self.model = model
        self.base_tmp_dir = base_tmp_dir or tempfile.gettempdir()
        self._memory_dir: Optional[str] = None
        self._phase_evidence_count: int = 0
        self._last_retrieved_context: str = ""
        self._answer_token_usage: Dict = {"input_tokens": 0, "output_tokens": 0}

    def reset(self):
        if self._memory_dir and os.path.isdir(self._memory_dir):
            shutil.rmtree(self._memory_dir, ignore_errors=True)
        ts = int(time.time() * 1000)
        self._memory_dir = os.path.join(
            self.base_tmp_dir, f"meme_evomem_{os.getpid()}_{ts}"
        )
        os.makedirs(self._memory_dir, exist_ok=True)
        _rebuild_memory_index(self._memory_dir)
        self._phase_evidence_count = 0
        self._last_retrieved_context = ""
        self._answer_token_usage = {"input_tokens": 0, "output_tokens": 0}

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest_session(self, session: dict) -> dict:
        if self._memory_dir is None:
            self.reset()

        if session.get("type") == "filler":
            return {
                "skipped": True,
                "reason": "filler session",
                "memory_entries": 0,
                "token_usage": {"input_tokens": 0, "output_tokens": 0},
            }

        session_text = _format_session(session)
        current_memory = _read_memory_files(self._memory_dir)

        if current_memory:
            user_prompt = (
                f"Current memory files:\n{current_memory}\n\n"
                f"New session to process:\n{session_text}\n\n"
                f"Update memory files based on the new session."
            )
        else:
            user_prompt = (
                f"New session to process:\n{session_text}\n\n"
                f"Create memory files for any useful information."
            )

        try:
            raw_output = _call_claude(user_prompt, INGEST_SYSTEM_PROMPT,
                                      self.model, cwd=self._memory_dir)
        except Exception as e:
            return {
                "error": str(e),
                "memory_entries": 0,
                "token_usage": {"input_tokens": 0, "output_tokens": 0},
            }

        session_id = session.get("session_id", f"s{int(time.time())}")
        files_written, files_deleted = _apply_file_changes(
            self._memory_dir, raw_output, session_id
        )
        _rebuild_memory_index(self._memory_dir)
        self._phase_evidence_count += 1

        return {
            "files_written": files_written,
            "files_deleted": files_deleted,
            "memory_entries": len(files_written),
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
        }

    # ------------------------------------------------------------------
    # Refine  (EvoMemory contribution — called by run_agent after each phase)
    # ------------------------------------------------------------------

    def finalize_ingest(self) -> None:
        """Refine pass: reorganize entire memory store for coherence."""
        if not self._memory_dir or self._phase_evidence_count == 0:
            self._phase_evidence_count = 0
            return

        current_memory = _read_memory_files(self._memory_dir)
        if not current_memory:
            self._phase_evidence_count = 0
            return

        user_prompt = (
            f"Current memory files to refine:\n\n{current_memory}\n\n"
            f"Produce the refined memory store."
        )

        try:
            raw_output = _call_claude(user_prompt, REFINE_SYSTEM_PROMPT,
                                      self.model, timeout=240,
                                      cwd=self._memory_dir)
        except Exception as e:
            print(f"      [evomem] Refine failed: {e}")
            self._phase_evidence_count = 0
            return

        files_written, files_deleted = _apply_file_changes(
            self._memory_dir, raw_output, "refine"
        )
        _rebuild_memory_index(self._memory_dir)
        self._phase_evidence_count = 0

        print(f"      [evomem] Refine: {len(files_written)} updated, "
              f"{len(files_deleted)} deleted")

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> str:
        """Return all memory file bodies concatenated (no LLM call)."""
        if not self._memory_dir or not os.path.isdir(self._memory_dir):
            self._last_retrieved_context = "(no memory)"
            return "(no memory)"

        parts = []
        for fname in sorted(os.listdir(self._memory_dir)):
            if fname == "MEMORY.md" or not fname.endswith(".md"):
                continue
            fpath = os.path.join(self._memory_dir, fname)
            try:
                with open(fpath) as f:
                    body = _strip_frontmatter(f.read())
                if body:
                    parts.append(body)
            except OSError:
                pass

        context = "\n\n".join(parts) if parts else "(no memory)"
        self._last_retrieved_context = context
        return context

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def get_memory_snapshot(self) -> dict:
        return {"text": _read_memory_files(self._memory_dir) or "(no memory)"}

    def get_retrieved_context(self) -> str:
        return self._last_retrieved_context
