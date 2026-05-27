"""
A-Mem agent for MeME evaluation.

Based on: "A-MEM: Agentic Memory for LLM Agents" (Xu et al., 2025)
Zettelkasten-inspired: atomic notes with keywords/tags/context + embedding links.

Architecture:
  ingest_session():  For each evidence session, two claude -p calls:
    1. Note construction — LLM extracts keywords, tags, contextual summary
    2. Link + evolve   — given top-k similar notes (by embedding), LLM
                         decides which to link AND updates their context
  retrieve():        Embed query → cosine similarity → top-k notes + linked
                     notes returned as context. No LLM call.
  answer_question(): Base class (retrieve → unified_llm).

Storage: one JSON file per note in {temp_dir}/notes/
Embeddings: all-MiniLM-L6-v2 (22MB, runs on CPU, ~1ms per encode)
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from typing import Dict, List, Optional

import numpy as np

from agents.base import BaseMemorySystem

# Lazy-loaded singleton to avoid loading the model on every episode
_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
    return _ENCODER


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

NOTE_CONSTRUCTION_PROMPT = """\
You are building a Zettelkasten-style memory note from a conversation session.

Extract the key information and produce a structured note with:
- keywords: 3-8 key terms (names, medications, locations, events)
- tags: 2-4 category tags (e.g. health, relationship, work, hobby, vehicle)
- context: a concise 1-3 sentence summary capturing the core facts

Output ONLY valid JSON — no prose, no fences. Start with {:
{"keywords": ["word1", "word2"], "tags": ["tag1"], "context": "summary of key facts"}
"""

LINK_AND_EVOLVE_PROMPT = """\
You are managing a Zettelkasten memory network. A new memory note has been created. \
Review it alongside existing related notes and:

1. LINKS: Decide which existing notes should be linked to the new note \
   (shared entities, related facts, or complementary information). \
   Return their IDs in "links".

2. EVOLVE: For each existing note that is linked, update its "context", "keywords", \
   and "tags" if the new note adds relevant information that should be reflected. \
   Only update if meaningfully improved — return the note ID and updated fields.

Output ONLY valid JSON — no prose, no fences. Start with {:
{
  "links": ["id1", "id2"],
  "evolved": [
    {"id": "id1", "context": "updated context", "keywords": ["kw1"], "tags": ["t1"]}
  ]
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


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _notes_dir(memory_dir: str) -> str:
    d = os.path.join(memory_dir, "notes")
    os.makedirs(d, exist_ok=True)
    return d


def _load_notes(memory_dir: str) -> Dict[str, dict]:
    nd = _notes_dir(memory_dir)
    notes = {}
    for fname in os.listdir(nd):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(nd, fname)) as f:
                note = json.load(f)
            notes[note["id"]] = note
        except Exception:
            pass
    return notes


def _save_note(memory_dir: str, note: dict) -> None:
    nd = _notes_dir(memory_dir)
    with open(os.path.join(nd, f"{note['id']}.json"), "w") as f:
        json.dump(note, f, ensure_ascii=False)


def _top_k_similar(query_emb: np.ndarray, notes: Dict[str, dict],
                   k: int = 5) -> List[str]:
    """Return IDs of top-k notes most similar to query_emb."""
    scored = []
    for nid, note in notes.items():
        emb = note.get("embedding")
        if emb is None:
            continue
        sim = _cosine_sim(query_emb, np.array(emb))
        scored.append((sim, nid))
    scored.sort(reverse=True)
    return [nid for _, nid in scored[:k]]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AMem(BaseMemorySystem):
    """
    A-Mem: Zettelkasten-style agentic memory using claude -p + embeddings.

    Each evidence session produces one atomic note with LLM-generated
    keywords/tags/context. Notes are linked by embedding similarity + LLM
    judgment. Linked notes are evolved (their context updated) when new
    related information arrives. Retrieval is pure embedding similarity —
    no LLM call at query time.
    """

    TOP_K_LINKS = 5   # candidates for link generation
    TOP_K_RETRIEVE = 5  # notes returned at retrieve time

    def __init__(self, model: str = "claude-code",
                 base_tmp_dir: Optional[str] = None):
        self.model = model
        self.base_tmp_dir = base_tmp_dir or tempfile.gettempdir()
        self._memory_dir: Optional[str] = None
        self._last_retrieved_context: str = ""
        self._answer_token_usage: Dict = {"input_tokens": 0, "output_tokens": 0}

    def reset(self):
        if self._memory_dir and os.path.isdir(self._memory_dir):
            shutil.rmtree(self._memory_dir, ignore_errors=True)
        ts = int(time.time() * 1000)
        self._memory_dir = os.path.join(
            self.base_tmp_dir, f"meme_amem_{os.getpid()}_{ts}"
        )
        os.makedirs(self._memory_dir, exist_ok=True)
        _notes_dir(self._memory_dir)  # create notes/ subdir
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
                "token_usage": {"input_tokens": 0, "output_tokens": 0},
            }

        session_text = _format_session(session)
        encoder = _get_encoder()

        # ---- Step 1: Note Construction ----
        try:
            raw = _call_claude(
                f"Session to process:\n{session_text}",
                NOTE_CONSTRUCTION_PROMPT,
                self.model,
                cwd=self._memory_dir,
            )
            parsed = json.loads(_extract_json(raw))
            keywords = parsed.get("keywords", [])
            tags = parsed.get("tags", [])
            context = parsed.get("context", session_text[:200])
        except Exception as e:
            # Fallback: minimal note with no LLM enrichment
            keywords, tags, context = [], [], session_text[:200]

        # Build embedding from enriched content
        embed_text = f"{context} {' '.join(keywords)} {' '.join(tags)}"
        embedding = encoder.encode([embed_text])[0].tolist()

        note_id = f"note_{uuid.uuid4().hex[:8]}"
        note = {
            "id": note_id,
            "content": session_text,
            "timestamp": session.get("timestamp", ""),
            "keywords": keywords,
            "tags": tags,
            "context": context,
            "embedding": embedding,
            "links": [],
        }

        # ---- Step 2: Link Generation + Memory Evolution ----
        all_notes = _load_notes(self._memory_dir)
        links_created = []
        evolved_ids = []

        if all_notes:
            # Find top-k similar existing notes
            note_emb = np.array(embedding)
            top_ids = _top_k_similar(note_emb, all_notes, k=self.TOP_K_LINKS)
            top_notes = {nid: all_notes[nid] for nid in top_ids if nid in all_notes}

            if top_notes:
                # Prepare compact representation of neighbors for the LLM
                neighbors_text = "\n\n".join(
                    f"ID: {nid}\nContext: {n['context']}\n"
                    f"Keywords: {', '.join(n.get('keywords', []))}\n"
                    f"Tags: {', '.join(n.get('tags', []))}"
                    for nid, n in top_notes.items()
                )
                new_note_text = (
                    f"NEW NOTE (id={note_id}):\n"
                    f"Context: {context}\n"
                    f"Keywords: {', '.join(keywords)}\n"
                    f"Tags: {', '.join(tags)}"
                )
                user_prompt = (
                    f"{new_note_text}\n\n"
                    f"EXISTING RELATED NOTES:\n{neighbors_text}"
                )

                try:
                    raw2 = _call_claude(
                        user_prompt, LINK_AND_EVOLVE_PROMPT,
                        self.model, cwd=self._memory_dir,
                    )
                    le = json.loads(_extract_json(raw2))
                    links_created = [lid for lid in le.get("links", [])
                                     if lid in all_notes]
                    note["links"] = links_created

                    # Apply evolutions to existing notes
                    for ev in le.get("evolved", []):
                        eid = ev.get("id")
                        if eid not in all_notes:
                            continue
                        existing = all_notes[eid]
                        if ev.get("context"):
                            existing["context"] = ev["context"]
                        if ev.get("keywords"):
                            existing["keywords"] = ev["keywords"]
                        if ev.get("tags"):
                            existing["tags"] = ev["tags"]
                        # Re-embed the evolved note
                        new_embed_text = (
                            f"{existing['context']} "
                            f"{' '.join(existing.get('keywords', []))} "
                            f"{' '.join(existing.get('tags', []))}"
                        )
                        existing["embedding"] = encoder.encode([new_embed_text])[0].tolist()
                        _save_note(self._memory_dir, existing)
                        evolved_ids.append(eid)
                except Exception:
                    pass  # Link/evolve failed — note is still saved without links

        _save_note(self._memory_dir, note)

        return {
            "note_id": note_id,
            "links_created": links_created,
            "evolved": evolved_ids,
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
        }

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> str:
        """Embed query → cosine similarity → top-k notes + their links. No LLM call."""
        if not self._memory_dir or not os.path.isdir(self._memory_dir):
            self._last_retrieved_context = "(no memory)"
            return "(no memory)"

        all_notes = _load_notes(self._memory_dir)
        if not all_notes:
            self._last_retrieved_context = "(no memory)"
            return "(no memory)"

        encoder = _get_encoder()
        query_emb = encoder.encode([question])[0]
        top_ids = _top_k_similar(query_emb, all_notes, k=self.TOP_K_RETRIEVE)

        # Include linked notes
        selected_ids = set(top_ids)
        for nid in top_ids:
            note = all_notes.get(nid, {})
            for linked_id in note.get("links", []):
                if linked_id in all_notes:
                    selected_ids.add(linked_id)

        parts = []
        for nid in sorted(selected_ids):
            note = all_notes.get(nid)
            if not note:
                continue
            kw = ", ".join(note.get("keywords", []))
            tags = ", ".join(note.get("tags", []))
            parts.append(
                f"[{note.get('timestamp', '')}] {note['context']}"
                + (f"\nKeywords: {kw}" if kw else "")
                + (f"\nTags: {tags}" if tags else "")
            )

        context = "\n\n".join(parts) if parts else "(no memory)"
        self._last_retrieved_context = context
        return context

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def get_memory_snapshot(self) -> dict:
        all_notes = _load_notes(self._memory_dir) if self._memory_dir else {}
        if not all_notes:
            return {"text": "(no memory)"}
        parts = []
        for note in sorted(all_notes.values(), key=lambda n: n.get("timestamp", "")):
            parts.append(
                f"[{note.get('timestamp','')}] {note['context']} "
                f"| kw: {', '.join(note.get('keywords',[]))} "
                f"| links: {note.get('links',[])}"
            )
        return {"text": "\n".join(parts)}

    def get_retrieved_context(self) -> str:
        return self._last_retrieved_context
