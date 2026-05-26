"""
Claude Code LLM Wiki memory system for MeME evaluation.

Implements the Karpathy LLM Wiki pattern using claude -p CLI.

Architecture:
  - Ingest: session → claude -p → structured wiki page updates (JSON)
            Filler sessions skipped. One page per entity/person/project.
  - Retrieve: question → claude -p picks relevant pages from index → read content
              (1 LLM call at retrieve time — unlike auto_memory which reads all files)
  - Answer: base class unified_llm uses retrieved context

Wiki structure per episode:
  {temp_dir}/
  ├── INDEX.md       — one line per page, rebuilt after each ingest
  └── pages/         — entity/concept/topic .md files

Key differentiator vs auto_memory:
  - Pages are entity-centric (one page per person/project) with cross-links
  - Attribute changes are timestamped; deletions marked "(discontinued)"
  - Retrieval is index-guided (LLM picks relevant pages), not dump-all
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


INGEST_SYSTEM_PROMPT = """\
You are a wiki editor maintaining a personal knowledge wiki. Read the new \
conversation session and update the wiki accordingly.

## Wiki page format

# Entity Name
*entity | concept | event*

- Attribute: Value (as of YYYY/MM/DD)
- Attribute: Old value → New value (updated YYYY/MM/DD)
- Attribute: Some thing (discontinued YYYY/MM/DD)

Related: [[Other Entity]], [[Another Page]]

## Rules
- One page per person, medication, project, or recurring concept
- Always show the CURRENT value of each attribute — use "→" for updates
- Mark removed/cancelled facts as "(discontinued DATE)" not deleted
- Cross-link related entities with [[Page Name]]
- Update existing pages; never create duplicates
- For sessions with no trackable facts, return empty pages list
- Keep page names short (snake_case, no spaces)

## Output format
Output ONLY valid JSON — no prose, no markdown fences, no explanation. Start with {:
{"pages": [{"name": "alice.md", "content": "# Alice\\n*entity*\\n\\n- ..."}], "delete": []}
"""

RETRIEVE_SYSTEM_PROMPT = """\
You are a wiki search assistant. Given a question and a wiki index, identify \
which pages are most likely to contain the answer.

Output ONLY a JSON list of filenames — no markdown, no explanation:
["filename1.md", "filename2.md"]

Return at most 5 relevant pages. Return [] if nothing is relevant.
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


def _read_index(wiki_dir: str) -> str:
    index_path = os.path.join(wiki_dir, "INDEX.md")
    if not os.path.exists(index_path):
        return "(empty index)"
    with open(index_path) as f:
        return f.read().strip() or "(empty index)"


def _rebuild_index(wiki_dir: str) -> None:
    pages_dir = os.path.join(wiki_dir, "pages")
    entries = []
    if os.path.isdir(pages_dir):
        for fname in sorted(os.listdir(pages_dir)):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(pages_dir, fname)
            title = fname[:-3]
            summary = ""
            try:
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("# "):
                            title = line[2:]
                        elif line and not line.startswith("*") and not line.startswith("Related:"):
                            summary = line[:100]
                            break
            except OSError:
                pass
            entries.append(f"- [{title}]({fname}): {summary}")

    index_path = os.path.join(wiki_dir, "INDEX.md")
    with open(index_path, "w") as f:
        if entries:
            f.write("# Wiki Index\n\n" + "\n".join(entries) + "\n")
        else:
            f.write("# Wiki Index\n\n(empty)\n")


def _read_all_pages(wiki_dir: str) -> str:
    pages_dir = os.path.join(wiki_dir, "pages")
    if not os.path.isdir(pages_dir):
        return ""
    parts = []
    for fname in sorted(os.listdir(pages_dir)):
        if not fname.endswith(".md"):
            continue
        try:
            with open(os.path.join(pages_dir, fname)) as f:
                content = f.read().strip()
            if content:
                parts.append(f"### {fname}\n{content}")
        except OSError:
            pass
    return "\n\n".join(parts)


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


def _extract_json(text: str) -> str:
    """Extract JSON from text that may contain prose, code fences, or both."""
    text = text.strip()
    # Try to find a ```json ... ``` or ``` ... ``` block anywhere in the text
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    # Fallback: find first { ... } span
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return text


def _apply_page_changes(wiki_dir: str, raw_output: str) -> tuple:
    pages_dir = os.path.join(wiki_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)

    pages_written, pages_deleted = [], []
    text = _extract_json(raw_output)

    try:
        parsed = json.loads(text)

        for pspec in parsed.get("pages", []):
            name = re.sub(r"[^\w\-.]", "_", (pspec.get("name") or "").strip())
            content = (pspec.get("content") or "").strip()
            if not name or not content:
                continue
            if not name.endswith(".md"):
                name += ".md"
            with open(os.path.join(pages_dir, name), "w") as f:
                f.write(content + "\n")
            pages_written.append(name)

        for name in parsed.get("delete", []):
            name = re.sub(r"[^\w\-.]", "_", (name or "").strip())
            fpath = os.path.join(pages_dir, name)
            if os.path.isfile(fpath):
                os.remove(fpath)
                pages_deleted.append(name)

    except (json.JSONDecodeError, KeyError):
        pass  # No wiki updates — session had no trackable facts

    return pages_written, pages_deleted


class ClaudeCodeWikiMemory(BaseMemorySystem):
    """
    LLM Wiki memory system using claude -p CLI (Karpathy pattern).

    Per-episode memory lives in a fresh temp directory.
    Evidence sessions are ingested via claude -p to update wiki pages.
    Filler sessions are skipped.
    Retrieval uses one LLM call to pick relevant pages from the index.
    """

    def __init__(self, model: str = "claude-code",
                 base_tmp_dir: Optional[str] = None):
        self.model = model
        self.base_tmp_dir = base_tmp_dir or tempfile.gettempdir()
        self._wiki_dir: Optional[str] = None
        self._last_retrieved_context: str = ""
        self._answer_token_usage: Dict = {"input_tokens": 0, "output_tokens": 0}

    def reset(self):
        if self._wiki_dir and os.path.isdir(self._wiki_dir):
            shutil.rmtree(self._wiki_dir, ignore_errors=True)
        ts = int(time.time() * 1000)
        self._wiki_dir = os.path.join(
            self.base_tmp_dir, f"meme_wiki_{os.getpid()}_{ts}"
        )
        os.makedirs(os.path.join(self._wiki_dir, "pages"), exist_ok=True)
        _rebuild_index(self._wiki_dir)
        self._last_retrieved_context = ""
        self._answer_token_usage = {"input_tokens": 0, "output_tokens": 0}

    def ingest_session(self, session: dict) -> dict:
        if self._wiki_dir is None:
            self.reset()

        if session.get("type") == "filler":
            return {
                "skipped": True,
                "reason": "filler session",
                "pages_written": 0,
                "token_usage": {"input_tokens": 0, "output_tokens": 0},
            }

        session_text = _format_session(session)
        current_pages = _read_all_pages(self._wiki_dir)
        current_index = _read_index(self._wiki_dir)

        if current_pages:
            user_prompt = (
                f"Current wiki index:\n{current_index}\n\n"
                f"Current wiki pages:\n{current_pages}\n\n"
                f"New session to process:\n{session_text}\n\n"
                f"Update the wiki based on the new session."
            )
        else:
            user_prompt = (
                f"New session to process:\n{session_text}\n\n"
                f"Create wiki pages for important people, projects, and tracked facts."
            )

        try:
            raw_output = _call_claude(user_prompt, INGEST_SYSTEM_PROMPT, self.model,
                                      cwd=self._wiki_dir)
        except Exception as e:
            return {
                "error": str(e),
                "pages_written": 0,
                "token_usage": {"input_tokens": 0, "output_tokens": 0},
            }

        pages_written, pages_deleted = _apply_page_changes(self._wiki_dir, raw_output)
        _rebuild_index(self._wiki_dir)

        return {
            "pages_written": pages_written,
            "pages_deleted": pages_deleted,
            "pages_written_count": len(pages_written),
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
        }

    def retrieve(self, question: str) -> str:
        """Index-guided retrieval: LLM picks relevant pages, then read them."""
        if not self._wiki_dir or not os.path.isdir(self._wiki_dir):
            self._last_retrieved_context = "(no wiki)"
            return "(no wiki)"

        pages_dir = os.path.join(self._wiki_dir, "pages")
        index = _read_index(self._wiki_dir)

        if index == "(empty index)" or not os.path.isdir(pages_dir):
            self._last_retrieved_context = "(no wiki content)"
            return "(no wiki content)"

        # Ask Claude which pages are relevant to this question
        user_prompt = f"Wiki index:\n{index}\n\nQuestion: {question}"
        try:
            raw = _call_claude(user_prompt, RETRIEVE_SYSTEM_PROMPT, self.model,
                               timeout=60, cwd=self._wiki_dir)
            relevant_pages: List[str] = json.loads(_extract_json(raw))
            if not isinstance(relevant_pages, list):
                raise ValueError("not a list")
        except Exception:
            # Fallback: return all pages
            relevant_pages = [f for f in os.listdir(pages_dir) if f.endswith(".md")]

        # Read the selected pages
        parts = []
        for fname in relevant_pages:
            fname = re.sub(r"[^\w\-.]", "_", (fname or "").strip())
            if not fname.endswith(".md"):
                fname += ".md"
            fpath = os.path.join(pages_dir, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath) as f:
                        parts.append(f.read().strip())
                except OSError:
                    pass

        context = "\n\n---\n\n".join(parts) if parts else "(no relevant wiki pages found)"
        self._last_retrieved_context = context
        return context

    def get_memory_snapshot(self) -> dict:
        if not self._wiki_dir:
            return {"text": "(no wiki)"}
        index = _read_index(self._wiki_dir)
        all_pages = _read_all_pages(self._wiki_dir)
        text = f"INDEX:\n{index}"
        if all_pages:
            text += f"\n\nPAGES:\n{all_pages}"
        return {"text": text}

    def get_retrieved_context(self) -> str:
        return self._last_retrieved_context
