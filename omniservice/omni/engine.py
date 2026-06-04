"""OmniEngine — the namespaced memory pipeline.

Wraps the OmniMemory design (raw archive -> background EXTRACT -> debounced VERIFY
-> ranked retrieve) as a long-running, multi-tenant engine. One instance serves
many namespaces; each namespace has its own storage dir, write lock, and VERIFY
debounce timer.
"""

import json
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional

from omni import config, prompts, storage
from omni.llm import call_local, extract_json


def _norm_entity(name: str) -> str:
    return re.sub(r"\s+", "_", (name or "").strip().lower())


def format_turns(turns: List[dict], timestamp: str = "") -> str:
    parts = [f"[Segment: {timestamp}]"] if timestamp else ["[Segment]"]
    for turn in turns:
        role = turn.get("role", "")
        role = "User" if role == "user" else ("Assistant" if role == "assistant" else role.capitalize())
        content = (turn.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _fmt_value(val) -> str:
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val) if val is not None else "?"


class _NamespaceState:
    def __init__(self, client_id: str, namespace: str, directory: str):
        self.client_id = client_id
        self.namespace = namespace
        self.dir = directory
        self.lock = threading.Lock()          # serializes state/deletions writes
        self.futures_lock = threading.Lock()
        self.futures: list = []               # in-flight EXTRACT futures
        self.verify_timer: Optional[threading.Timer] = None
        self.verify_lock = threading.Lock()
        # Single worker per namespace => EXTRACT runs in FIFO ingest order, so a
        # delete never races ahead of the create it supersedes. Cross-namespace
        # work still runs concurrently (one worker thread each).
        self.executor = ThreadPoolExecutor(max_workers=1)


class OmniEngine:
    def __init__(self,
                 extract_model: str = None,
                 verify_model: str = None,
                 debounce_seconds: float = None,
                 workers: int = None):
        self.extract_model = extract_model or config.EXTRACT_MODEL
        self.verify_model = verify_model or config.VERIFY_MODEL
        self.debounce_seconds = debounce_seconds if debounce_seconds is not None else config.VERIFY_DEBOUNCE_SECONDS
        self._ns: Dict[tuple, _NamespaceState] = {}
        self._ns_lock = threading.Lock()

    # ------------------------------------------------------------------
    def _state(self, client_id: str, namespace: str) -> _NamespaceState:
        key = (client_id, namespace)
        with self._ns_lock:
            st = self._ns.get(key)
            if st is None:
                st = _NamespaceState(client_id, namespace, storage.ns_dir(client_id, namespace))
                self._ns[key] = st
            return st

    # ------------------------------------------------------------------
    # Memory-op instrumentation
    # ------------------------------------------------------------------
    def _call_logged(self, model: str, prompt: str, system: str, max_tokens: int = 1024):
        """Run a local-LLM call; return (raw_output, duration_ms, error_or_None)."""
        t0 = time.time()
        raw, error = "", None
        try:
            raw = call_local(prompt, system, model=model, max_tokens=max_tokens)
        except Exception as e:
            error = str(e)
        return raw, int((time.time() - t0) * 1000), error

    def _log_action(self, st: _NamespaceState, op: str, model: str, prompt: str,
                    system: str, raw: str, parsed, applied,
                    duration_ms: int, error: Optional[str],
                    max_tokens: int = 1024, temperature: float = 0.0) -> None:
        storage.log_action(st.dir, {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "op": op,
            "model": model,
            "client_id": st.client_id,
            "namespace": st.namespace,
            # Full LLM request: system + user messages + sampling params.
            "system": system,
            "input": prompt,
            "params": {"temperature": temperature, "max_tokens": max_tokens},
            "output": raw,
            "parsed": parsed,
            "applied": applied,
            "duration_ms": duration_ms,
            "error": error,
        })

    def _log_event(self, st: _NamespaceState, op: str, input_text: str,
                   output: str = "", applied=None, duration_ms: int = 0) -> None:
        """Log a non-LLM API event (INGEST / RETRIEVE) into the same action timeline."""
        storage.log_action(st.dir, {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "op": op,
            "model": "",
            "client_id": st.client_id,
            "namespace": st.namespace,
            "system": "",
            "input": input_text,
            "params": {},
            "output": output,
            "parsed": None,
            "applied": applied or {},
            "duration_ms": duration_ms,
            "error": None,
        })

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def ingest(self, client_id: str, namespace: str, turns: List[dict],
               timestamp: str = "", source: str = "") -> dict:
        st = self._state(client_id, namespace)
        text = format_turns(turns, timestamp)
        raw_path = storage.archive_raw(st.dir, text)

        self._log_event(st, "INGEST", text, applied={
            "turns": len(turns), "source": source,
            "archived": os.path.basename(raw_path), "extract_queued": True,
        })

        fut = st.executor.submit(self._extract_and_update, st, text, timestamp)
        with st.futures_lock:
            st.futures.append(fut)

        self._schedule_verify(st)
        return {"archived": raw_path, "extract_queued": True}

    def _extract_and_update(self, st: _NamespaceState, text: str, timestamp: str) -> None:
        d = st.dir
        with st.lock:
            state = storage.load_state(d)
        state_summary = "\n".join(
            f'  "{k}": {json.dumps(v.get("current_value"))} '
            f'(type={v.get("type","scalar")}, as of {v.get("last_updated","?")})'
            for k, v in sorted(state.items())
        ) or "  (no entities yet)"
        prompt = f"Known entities:\n{state_summary}\n\nText to process:\n{text}"

        raw, dur, error = self._call_logged(self.extract_model, prompt, prompts.EXTRACT_SYSTEM)
        updates = []
        if not error:
            try:
                updates = json.loads(extract_json(raw)).get("updates", [])
            except Exception as e:
                error = f"parse error: {e}"

        applied: List[dict] = []
        ts = (timestamp or "")[:10]
        new_entities: List[str] = []

        if updates:
            with st.lock:
                state = storage.load_state(d)
                deletions = storage.load_deletions(d)

                for upd in updates:
                    entity = _norm_entity(upd.get("entity"))
                    if not entity:
                        continue
                    new_val = upd.get("new_value")
                    deleted = bool(upd.get("deleted", False))
                    upd_ts = (upd.get("timestamp") or ts)[:10]
                    etype = upd.get("type", "scalar")

                    old_info = state.get(entity, {})
                    old_val = old_info.get("current_value")
                    is_new = entity not in state

                    if deleted:
                        deletions[entity] = {
                            "was": old_val if old_val is not None else old_info.get("current_value", "?"),
                            "when": upd_ts,
                            "tombstone": f"Explicitly removed as of {upd_ts}. No replacement.",
                            "source": "extract",
                        }
                        state.pop(entity, None)
                    else:
                        if etype == "list" and isinstance(new_val, list):
                            existing_list = old_val if isinstance(old_val, list) else []
                            new_val = list(dict.fromkeys(existing_list + new_val))
                        state[entity] = {
                            "current_value": new_val,
                            "type": etype,
                            "first_seen": old_info.get("first_seen", upd_ts),
                            "last_updated": upd_ts,
                            "update_source": "extract",
                        }
                        # Re-asserting a value resurrects a previously deleted entity:
                        # keep state and deletions mutually exclusive.
                        deletions.pop(entity, None)
                        if is_new:
                            new_entities.append(entity)

                    applied.append({"entity": entity,
                                    "action": "deleted" if deleted else "updated",
                                    "value": None if deleted else new_val})
                    storage.append_history(d, {
                        "entity": entity, "old": old_val,
                        "new": new_val if not deleted else None,
                        "ts": upd_ts, "deleted": deleted, "source": "extract",
                    })

                storage.save_state(d, state)
                storage.save_deletions(d, deletions)

                for upd in updates:
                    if upd.get("deleted"):
                        continue
                    entity = _norm_entity(upd.get("entity"))
                    if entity and entity in state:
                        storage.write_page(d, entity, state)

        self._log_action(st, "EXTRACT", self.extract_model, prompt, prompts.EXTRACT_SYSTEM,
                          raw, {"updates": updates}, applied, dur, error)

        if new_entities and len(state) > len(new_entities):
            self._relate_new_entities(st, new_entities)

    def _relate_new_entities(self, st: _NamespaceState, new_entities: List[str]) -> None:
        d = st.dir
        with st.lock:
            state = storage.load_state(d)
        existing = {e: state[e] for e in state if e not in new_entities}
        if not existing:
            return
        new_summary = "\n".join(
            f'  {e}: {json.dumps(state[e].get("current_value"))}'
            for e in new_entities if e in state
        )
        existing_summary = "\n".join(
            f'  {e}: {json.dumps(info.get("current_value"))}'
            for e, info in sorted(existing.items())
        )
        prompt = f"New entities:\n{new_summary}\n\nExisting entities:\n{existing_summary}"
        raw, dur, error = self._call_logged(self.extract_model, prompt,
                                            prompts.RELATE_SYSTEM, max_tokens=512)
        edges = []
        if not error:
            try:
                edges = json.loads(extract_json(raw)).get("edges", [])
            except Exception as e:
                error = f"parse error: {e}"
        if edges:
            with st.lock:
                state = storage.load_state(d)
                touched = set(new_entities)
                for e in edges:
                    touched.add(_norm_entity(e.get("from")))
                    touched.add(_norm_entity(e.get("to")))
                for entity in touched:
                    if entity in state:
                        storage.write_page(d, entity, state, edges)
        self._log_action(st, "RELATE", self.extract_model, prompt, prompts.RELATE_SYSTEM,
                          raw, {"edges": edges}, {"edges": len(edges)}, dur, error, max_tokens=512)

    # ------------------------------------------------------------------
    # Debounced VERIFY
    # ------------------------------------------------------------------
    def _schedule_verify(self, st: _NamespaceState) -> None:
        with st.verify_lock:
            if st.verify_timer is not None:
                st.verify_timer.cancel()
            st.verify_timer = threading.Timer(self.debounce_seconds, self._run_verify, args=(st,))
            st.verify_timer.daemon = True
            st.verify_timer.start()

    def verify_now(self, client_id: str, namespace: str) -> dict:
        """Force a synchronous VERIFY (used by tests and explicit flush)."""
        st = self._state(client_id, namespace)
        with st.verify_lock:
            if st.verify_timer is not None:
                st.verify_timer.cancel()
                st.verify_timer = None
        return self._run_verify(st, wait=True)

    def _wait_pending(self, st: _NamespaceState, timeout: float = 120) -> None:
        with st.futures_lock:
            pending = list(st.futures)
            st.futures = []
        for f in pending:
            try:
                f.result(timeout=timeout)
            except Exception:
                pass

    def _run_verify(self, st: _NamespaceState, wait: bool = True) -> dict:
        d = st.dir
        if wait:
            self._wait_pending(st)

        state = storage.load_state(d)
        deletions = storage.load_deletions(d)
        if not state and not deletions:
            return {"applied": 0}

        sessions_block = "\n\n---\n\n".join(storage.read_recent_raw(d, config.VERIFY_RAW_WINDOW))
        if not sessions_block.strip():
            return {"applied": 0}

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
            f"## Transcripts\n\n{sessions_block}\n\n"
            f"## Current entity state\n{state_summary}\n\n"
            f"## Current deletions\n{deletions_summary}"
        )
        raw, dur, error = self._call_logged(self.verify_model, prompt, prompts.VERIFY_SYSTEM)
        parsed = {}
        if not error:
            try:
                parsed = json.loads(extract_json(raw))
            except Exception as e:
                error = f"parse error: {e}"

        corrections = parsed.get("corrections", []) if isinstance(parsed, dict) else []
        missed_dels = parsed.get("missed_deletions", []) if isinstance(parsed, dict) else []
        list_corrs = parsed.get("list_corrections", []) if isinstance(parsed, dict) else []
        if error or not (corrections or missed_dels or list_corrs):
            self._log_action(st, "VERIFY", self.verify_model, prompt, prompts.VERIFY_SYSTEM,
                             raw, parsed, {"applied": 0}, dur, error)
            return {"applied": 0}

        ts_now = time.strftime("%Y/%m/%d")
        with st.lock:
            state = storage.load_state(d)
            deletions = storage.load_deletions(d)

            for c in corrections:
                entity = _norm_entity(c.get("entity"))
                correct = c.get("correct_value")
                if not entity or correct is None:
                    continue
                old_info = state.get(entity, {})
                state[entity] = {
                    "current_value": correct,
                    "type": old_info.get("type", "scalar"),
                    "first_seen": old_info.get("first_seen", ts_now),
                    "last_updated": ts_now,
                    "update_source": "verified",
                }
                storage.append_history(d, {"entity": entity, "old": c.get("was"),
                                           "new": correct, "ts": ts_now,
                                           "deleted": False, "source": "verified"})
                storage.write_page(d, entity, state)

            for lc in list_corrs:
                entity = _norm_entity(lc.get("entity"))
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
                storage.append_history(d, {"entity": entity, "old": old_info.get("current_value"),
                                           "new": cor_list, "ts": ts_now,
                                           "deleted": False, "source": "verified"})
                storage.write_page(d, entity, state)

            for dd in missed_dels:
                entity = _norm_entity(dd.get("entity"))
                if not entity:
                    continue
                old_val = state.pop(entity, {}).get("current_value", dd.get("was"))
                deletions[entity] = {
                    "was": old_val, "when": ts_now,
                    "tombstone": f"Explicitly removed as of {ts_now}. No replacement.",
                    "source": "verified",
                }
                storage.append_history(d, {"entity": entity, "old": old_val, "new": None,
                                           "ts": ts_now, "deleted": True, "source": "verified"})

            storage.save_state(d, state)
            storage.save_deletions(d, deletions)

        applied = {"applied": len(corrections) + len(missed_dels) + len(list_corrs),
                   "corrections": len(corrections), "deletions": len(missed_dels),
                   "lists": len(list_corrs)}
        self._log_action(st, "VERIFY", self.verify_model, prompt, prompts.VERIFY_SYSTEM,
                         raw, parsed, applied, dur, error)
        return applied

    # ------------------------------------------------------------------
    # Retrieve  (always-include + ranked search; no LLM call)
    # ------------------------------------------------------------------
    @staticmethod
    def _tokens(text: str) -> set:
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    def retrieve(self, client_id: str, namespace: str, query: str = "", mode: str = "search") -> str:
        st = self._state(client_id, namespace)
        d = st.dir
        t0 = time.time()
        deletions = storage.load_deletions(d)
        state = storage.load_state(d)

        parts: List[str] = []

        # 1. Deletions — always included, prominent.
        if deletions:
            lines = ["## DELETED — No current value",
                     "(For these: answer 'was deleted' / 'no longer applicable' / 'I don't know')\n"]
            for entity, info in sorted(deletions.items()):
                lines.append(f"- **{entity}**: was {info.get('was','?')} "
                             f"(removed {info.get('when','?')}). {info.get('tombstone','No replacement.')}")
            parts.append("\n".join(lines))

        # 2. Current entity state — always included (bounded, drives ER/Cas).
        #    A tombstoned entity must never appear as a current value.
        live = {e: i for e, i in state.items() if e not in deletions}
        if live:
            lines = ["## Current Entity State\n"]
            for entity, info in sorted(live.items()):
                tag = " [verified]" if info.get("update_source") == "verified" else ""
                lines.append(f"- **{entity}**: {_fmt_value(info.get('current_value'))} "
                             f"(as of {info.get('last_updated','?')}){tag}")
            parts.append("\n".join(lines))

        if mode == "session-start":
            context = "\n\n".join(parts) if parts else "(no memory)"
            self._log_event(st, "RETRIEVE", f"({mode})", output=context,
                            applied={"mode": mode, "chars": len(context)},
                            duration_ms=int((time.time() - t0) * 1000))
            return context

        # 3. Ranked entity pages by query overlap (Agg / relationships).
        qtokens = self._tokens(query)
        pages = storage.all_pages(d)
        if pages:
            scored = []
            for entity, content in pages.items():
                etoks = self._tokens(entity) | self._tokens(content)
                score = len(qtokens & etoks) if qtokens else 0
                scored.append((score, entity, content))
            scored.sort(key=lambda x: (-x[0], x[1]))
            top = [c for s, e, c in scored[:config.RETRIEVE_TOP_PAGES] if (s > 0 or not qtokens)]
            if top:
                parts.append("## Entity Relationships\n\n" + "\n\n---\n\n".join(c.strip() for c in top))

        # 4. Ranked revision history (Tr).
        history = storage.read_history(d)
        if history:
            if qtokens:
                scored_h = []
                for ev in history:
                    htoks = self._tokens(ev.get("entity", "")) | self._tokens(_fmt_value(ev.get("new"))) | self._tokens(_fmt_value(ev.get("old")))
                    scored_h.append((len(qtokens & htoks), ev))
                scored_h.sort(key=lambda x: -x[0])
                hist = [ev for s, ev in scored_h[:config.RETRIEVE_TOP_HISTORY] if s > 0] \
                    or history[-config.RETRIEVE_TOP_HISTORY:]
            else:
                hist = history[-config.RETRIEVE_TOP_HISTORY:]
            lines = ["## Revision History\n"]
            for ev in hist:
                e, old, new = ev.get("entity", "?"), ev.get("old"), ev.get("new")
                tsv, src = ev.get("ts", "?"), ev.get("source", "")
                tag = " [verified]" if src == "verified" else ""
                if ev.get("deleted"):
                    lines.append(f"[{tsv}] {e}: DELETED (was: {_fmt_value(old)}){tag}")
                elif old is None:
                    lines.append(f"[{tsv}] {e}: -> {_fmt_value(new)}{tag}")
                else:
                    lines.append(f"[{tsv}] {e}: {_fmt_value(old)} -> {_fmt_value(new)}{tag}")
            parts.append("\n".join(lines))

        # 5. Recent summaries.
        summaries = storage.read_summaries(d, 4)
        if summaries:
            lines = ["## Recent Context\n"]
            for s in summaries:
                lines.append(f"  [{s.get('ts','')}] {s.get('summary','')}")
            parts.append("\n".join(lines))

        context = "\n\n".join(parts) if parts else "(no memory)"
        # Log real client queries; skip mode="full" (internal snapshot / UI inspection).
        if mode != "full":
            self._log_event(st, "RETRIEVE", query or "(empty query)", output=context,
                            applied={"mode": mode, "chars": len(context)},
                            duration_ms=int((time.time() - t0) * 1000))
        return context

    # ------------------------------------------------------------------
    def snapshot(self, client_id: str, namespace: str) -> str:
        return self.retrieve(client_id, namespace, query="", mode="full")

    def reset(self, client_id: str, namespace: str) -> None:
        st = self._state(client_id, namespace)
        with st.verify_lock:
            if st.verify_timer is not None:
                st.verify_timer.cancel()
                st.verify_timer = None
        self._wait_pending(st)
        st.executor.shutdown(wait=False)
        if st.dir and st.dir.startswith(str(config.STORAGE_ROOT)):
            shutil.rmtree(st.dir, ignore_errors=True)
        with self._ns_lock:
            self._ns.pop((client_id, namespace), None)
        storage.ns_dir(client_id, namespace)  # recreate empty

    def reset_client(self, client_id: str) -> dict:
        """Wipe an entire client (all its namespaces). Used for test teardown."""
        namespaces = storage.list_namespaces(client_id)
        with self._ns_lock:
            keys = [k for k in self._ns if k[0] == client_id]
            states = [self._ns.pop(k) for k in keys]
        for st in states:
            with st.verify_lock:
                if st.verify_timer is not None:
                    st.verify_timer.cancel()
                    st.verify_timer = None
            self._wait_pending(st)
            st.executor.shutdown(wait=False)
        cdir = storage.client_dir(client_id)
        if cdir and cdir.startswith(str(config.STORAGE_ROOT)) and cdir != str(config.STORAGE_ROOT):
            shutil.rmtree(cdir, ignore_errors=True)
        return {"client_id": client_id, "namespaces_removed": len(namespaces)}
