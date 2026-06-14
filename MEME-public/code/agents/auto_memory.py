"""
Claude Code Auto-Memory agent for MeME evaluation.

Implements the BaseMemorySystem interface using Claude Code's native auto-memory
format (MEMORY.md index + typed .md files with frontmatter).

ingest_session(): Calls `claude -p` to decide what to write to memory files.
                  Filler sessions are skipped (they contain no tracked facts).
retrieve():       Reads memory files directly (no LLM call — mirrors Claude Code
                  behavior where files are loaded at session start).
answer_question(): Uses base class (retrieve → unified_llm).
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Dict, Optional

from agents.base import BaseMemorySystem


INGEST_SYSTEM_PROMPT = """\
You are Claude Code's auto-memory system. Read a new conversation session and \
decide what information is worth saving to persistent memory files.

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
Output ONLY valid JSON with no markdown fences or explanation:
{"files": [{"name": "filename.md", "content": "---\\nname: ...\\n...\\n---\\n\\nbody"}], "delete": []}
"""


# "Dreaming" consolidation pass — matches Anthropic's "Dreams" feature
# (platform.claude.com/docs/en/managed-agents/dreams): it takes the memory store PLUS the
# raw session transcripts and "mines them for patterns and insights to fold into the
# output," producing a reorganized memory. Re-reading the source RECOVERS facts the lossy
# incremental writes missed. General consolidation only — no benchmark-specific
# instructions (e.g. no Abs uncertainty tell), so Abs/Cas gains reflect emergent behavior.
DREAM_SYSTEM_PROMPT = """\
You are performing a memory consolidation ("dreaming") pass. You are given the agent's \
CURRENT memory files AND the raw session transcripts they were built from. Produce a \
cleaner, more complete, consolidated set of memory files.

Goals:
- Merge duplicate or closely related facts split across files; remove redundancy.
- RECOVER from the transcripts any durable fact that is missing from the current memory \
or was recorded wrong — re-read the transcripts closely; incremental writes are lossy.
- Keep EVERY distinct durable fact. For a fact that changed over time, keep the CURRENT \
value (use transcript dates). If the user described a dependency and its trigger has \
since changed, update the dependent fact to its resulting current value.
- If a fact was explicitly removed/cancelled/ended, record it as removed; do NOT present \
its old value as current.
- Group related facts under well-named topic files (short snake_case.md, with frontmatter).

Output ONLY valid JSON with no fences:
{"files": [{"name": "filename.md", "content": "---\\nname: ...\\n---\\n\\nbody"}]}
Return the COMPLETE consolidated set of files — this REPLACES all current topic files."""


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


def _apply_file_changes(memory_dir: str, raw_output: str, session_id: str) -> tuple:
    """Parse claude output and write/delete memory files. Returns (written, deleted)."""
    files_written, files_deleted = [], []
    text = raw_output.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())

    try:
        parsed = json.loads(text)

        for fspec in parsed.get("files", []):
            fname = (fspec.get("name") or "").strip()
            content = (fspec.get("content") or "").strip()
            if not fname or not content:
                continue
            fname = re.sub(r"[^\w\-.]", "_", fname)
            if not fname.endswith(".md"):
                fname += ".md"
            with open(os.path.join(memory_dir, fname), "w") as f:
                f.write(content + "\n")
            files_written.append(fname)

        for fname in parsed.get("delete", []):
            fname = (fname or "").strip()
            fpath = os.path.join(memory_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
                files_deleted.append(fname)

    except (json.JSONDecodeError, KeyError):
        # Non-JSON: save raw output as a fallback memory entry
        if text:
            safe_id = re.sub(r'\W', '_', session_id)[:20]
            fname = f"notes_{safe_id}.md"
            with open(os.path.join(memory_dir, fname), "w") as f:
                slug = fname[:-3]
                f.write(
                    f"---\nname: {slug}\ndescription: session notes\n"
                    f"metadata:\n  type: project\n---\n\n{text}\n"
                )
            files_written.append(fname)

    return files_written, files_deleted


class ClaudeCodeAutoMemory(BaseMemorySystem):
    """
    MeME agent implementing Claude Code's auto-memory format.

    Per-episode memory lives in a fresh temp directory. Evidence sessions are
    ingested via `claude -p`; filler sessions are skipped (they contain no
    tracked facts). Retrieval reads files directly — no extra LLM call.
    """

    def __init__(self, model: str = "claude-code",
                 base_tmp_dir: Optional[str] = None, dreaming: bool = False):
        self.model = model
        self.base_tmp_dir = base_tmp_dir or tempfile.gettempdir()
        self.dreaming = dreaming   # Dreams-style consolidation over memory + transcripts
        self._memory_dir: Optional[str] = None
        self._transcripts: list = []   # archived evidence transcripts (for Dreams-style mining)
        self._last_retrieved_context: str = ""
        self._answer_token_usage: Dict = {"input_tokens": 0, "output_tokens": 0}

    def reset(self):
        if self._memory_dir and os.path.isdir(self._memory_dir):
            shutil.rmtree(self._memory_dir, ignore_errors=True)
        ts = int(time.time() * 1000)
        self._memory_dir = os.path.join(
            self.base_tmp_dir, f"meme_automem_{os.getpid()}_{ts}"
        )
        os.makedirs(self._memory_dir, exist_ok=True)
        _rebuild_memory_index(self._memory_dir)
        self._transcripts = []
        self._last_retrieved_context = ""
        self._answer_token_usage = {"input_tokens": 0, "output_tokens": 0}

    def ingest_session(self, session: dict) -> dict:
        if self._memory_dir is None:
            self.reset()

        # Skip filler sessions — they contain no tracked facts
        if session.get("type") == "filler":
            return {
                "skipped": True,
                "reason": "filler session",
                "memory_entries": 0,
                "token_usage": {"input_tokens": 0, "output_tokens": 0},
            }

        session_text = _format_session(session)
        # Archive the evidence transcript so the Dreams-style pass can re-mine the source.
        if self.dreaming:
            self._transcripts.append(session_text)
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
            raw_output = _call_claude(user_prompt, INGEST_SYSTEM_PROMPT, self.model,
                                      cwd=self._memory_dir)
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

        return {
            "files_written": files_written,
            "files_deleted": files_deleted,
            "memory_entries": len(files_written),
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
        }

    def finalize_ingest(self):
        """Dreaming consolidation pass over ALL memory files (once per phase).

        No-op unless dreaming is enabled. On a parse failure the existing memory is
        left untouched (a bad dream must never destroy the awake-written memory)."""
        if not self.dreaming or self._memory_dir is None:
            return
        current = _read_memory_files(self._memory_dir)
        if not current.strip():
            return

        # Dreams-style: feed memory + the raw source transcripts so the pass can recover
        # missed facts, not just reorganize what's already in memory.
        transcripts = "\n\n".join(
            f"--- Transcript {i+1} ---\n{t}" for i, t in enumerate(self._transcripts)
        )
        prompt = (
            f"Current memory files:\n{current}\n\n"
            f"Raw session transcripts (the source these were built from):\n{transcripts}\n\n"
            f"Consolidate AND recover any missed/incorrect facts from the transcripts; "
            f"return the complete file list."
        )
        try:
            raw = _call_claude(prompt, DREAM_SYSTEM_PROMPT, self.model, cwd=self._memory_dir)
        except Exception:
            return

        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text.strip())
        try:
            parsed = json.loads(text)
            files = parsed.get("files") or []
        except (json.JSONDecodeError, ValueError, AttributeError):
            return
        files = [f for f in files if (f.get("name") or "").strip()
                 and (f.get("content") or "").strip()]
        if not files:
            return  # nothing valid — keep existing memory

        # Replace the topic files with the consolidated set (keep MEMORY.md index).
        for fname in os.listdir(self._memory_dir):
            if fname != "MEMORY.md" and fname.endswith(".md"):
                try:
                    os.remove(os.path.join(self._memory_dir, fname))
                except OSError:
                    pass
        for fspec in files:
            fname = re.sub(r"[^\w\-.]", "_", fspec["name"].strip())
            if not fname.endswith(".md"):
                fname += ".md"
            with open(os.path.join(self._memory_dir, fname), "w") as f:
                f.write(fspec["content"].strip() + "\n")
        _rebuild_memory_index(self._memory_dir)

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

    def get_memory_snapshot(self) -> dict:
        return {"text": _read_memory_files(self._memory_dir) or "(no memory)"}

    def get_retrieved_context(self) -> str:
        return self._last_retrieved_context
