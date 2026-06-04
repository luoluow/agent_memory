"""Per-namespace on-disk storage.

Layout (one directory per namespace under config.STORAGE_ROOT):

    <ns>/
      raw/ingest_NNNN.txt   verbatim archived segments (never modified)
      state.json            current entity values (scalar + list)
      history.jsonl         append-only revision log
      deletions.json        explicit deletion ledger
      summaries.jsonl       compressed context
      pages/{entity}.md     KG-style wiki pages with [[links]]
      cursor.json           per-source transcript cursors + ingest counter
"""

import hashlib
import json
import os
import re
import threading
from typing import Any, Dict, List, Optional

from omni import config


# ---------------------------------------------------------------------------
# Namespace resolution
# ---------------------------------------------------------------------------

def _slug(value: str) -> str:
    slug = re.sub(r"[^\w.-]", "-", value or "").strip("-") or "default"
    if len(slug) > 80:
        slug = slug[:40] + "-" + hashlib.sha1(value.encode()).hexdigest()[:12]
    return slug


def client_slug(client_id: str) -> str:
    return _slug(client_id)


def ns_slug(namespace: str) -> str:
    return _slug(namespace)


def client_dir(client_id: str) -> str:
    d = os.path.join(str(config.STORAGE_ROOT), client_slug(client_id))
    os.makedirs(d, exist_ok=True)
    return d


def ns_dir(client_id: str, namespace: str) -> str:
    d = os.path.join(client_dir(client_id), ns_slug(namespace))
    os.makedirs(os.path.join(d, "raw"), exist_ok=True)
    os.makedirs(os.path.join(d, "pages"), exist_ok=True)
    return d


def list_clients() -> List[str]:
    root = str(config.STORAGE_ROOT)
    if not os.path.isdir(root):
        return []
    return sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))


def list_namespaces(client_id: str) -> List[str]:
    cdir = os.path.join(str(config.STORAGE_ROOT), client_slug(client_id))
    if not os.path.isdir(cdir):
        return []
    return sorted(n for n in os.listdir(cdir) if os.path.isdir(os.path.join(cdir, n)))


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _raw_dir(d: str) -> str:
    p = os.path.join(d, "raw")
    os.makedirs(p, exist_ok=True)
    return p


def _pages_dir(d: str) -> str:
    p = os.path.join(d, "pages")
    os.makedirs(p, exist_ok=True)
    return p


def _state_path(d: str) -> str:
    return os.path.join(d, "state.json")


def _history_path(d: str) -> str:
    return os.path.join(d, "history.jsonl")


def _deletions_path(d: str) -> str:
    return os.path.join(d, "deletions.json")


def _summaries_path(d: str) -> str:
    return os.path.join(d, "summaries.jsonl")


def _cursor_path(d: str) -> str:
    return os.path.join(d, "cursor.json")


def _actions_path(d: str) -> str:
    return os.path.join(d, "actions.jsonl")


# ---------------------------------------------------------------------------
# Raw archive
# ---------------------------------------------------------------------------

def archive_raw(d: str, text: str) -> str:
    """Write a verbatim segment to raw/ingest_NNNN.txt. Returns the path."""
    raw = _raw_dir(d)
    n = len([f for f in os.listdir(raw) if f.endswith(".txt")]) + 1
    path = os.path.join(raw, f"ingest_{n:04d}.txt")
    with open(path, "w") as f:
        f.write(text)
    return path


def read_recent_raw(d: str, n: int) -> List[str]:
    raw = _raw_dir(d)
    files = sorted(f for f in os.listdir(raw) if f.endswith(".txt"))
    texts = []
    for fname in files[-n:]:
        with open(os.path.join(raw, fname)) as f:
            texts.append(f.read())
    return texts


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def load_state(d: str) -> Dict[str, dict]:
    path = _state_path(d)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_state(d: str, state: Dict[str, dict]) -> None:
    tmp = _state_path(d) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _state_path(d))


# ---------------------------------------------------------------------------
# Deletions
# ---------------------------------------------------------------------------

def load_deletions(d: str) -> Dict[str, dict]:
    path = _deletions_path(d)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_deletions(d: str, deletions: Dict[str, dict]) -> None:
    tmp = _deletions_path(d) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(deletions, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _deletions_path(d))


# ---------------------------------------------------------------------------
# History (append-only)
# ---------------------------------------------------------------------------

def append_history(d: str, event: dict) -> None:
    with open(_history_path(d), "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_history(d: str, n: Optional[int] = None) -> List[dict]:
    path = _history_path(d)
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
    return events[-n:] if n else events


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def append_summary(d: str, summary: str, ts: str) -> None:
    with open(_summaries_path(d), "a") as f:
        f.write(json.dumps({"ts": ts, "summary": summary}) + "\n")


def read_summaries(d: str, n: int = 4) -> List[dict]:
    path = _summaries_path(d)
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


# ---------------------------------------------------------------------------
# Action log (audit trail of every memory-op LLM call)
# ---------------------------------------------------------------------------

def log_action(d: str, record: dict) -> None:
    with open(_actions_path(d), "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_actions(d: str, limit: Optional[int] = None) -> List[dict]:
    path = _actions_path(d)
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
    return items[-limit:] if limit else items


# ---------------------------------------------------------------------------
# Cursor (transcript dedup + ingest counter)
# ---------------------------------------------------------------------------

def load_cursor(d: str) -> dict:
    path = _cursor_path(d)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_cursor(d: str, cursor: dict) -> None:
    tmp = _cursor_path(d) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cursor, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _cursor_path(d))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_path(d: str, entity: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", entity)
    return os.path.join(_pages_dir(d), f"{safe}.md")


def load_page(d: str, entity: str) -> str:
    path = page_path(d, entity)
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read()


def all_pages(d: str) -> Dict[str, str]:
    pages = _pages_dir(d)
    out = {}
    for fname in os.listdir(pages):
        if fname.endswith(".md"):
            with open(os.path.join(pages, fname)) as f:
                out[fname[:-3]] = f.read()
    return out


def write_page(d: str, entity: str, state: Dict[str, dict],
               edges: Optional[List[dict]] = None) -> None:
    info = state.get(entity, {})
    val = info.get("current_value", "?")
    ts = info.get("last_updated", "?")
    src = info.get("update_source", "extract")

    existing = load_page(d, entity)
    existing_edges: List[str] = []
    history_block = ""
    if existing:
        m = re.search(r"## Relationships\n(.*?)(?=\n##|\Z)", existing, re.DOTALL)
        if m:
            existing_edges = [l.strip() for l in m.group(1).splitlines() if l.strip()]
        m = re.search(r"## History\n(.*)", existing, re.DOTALL)
        if m:
            history_block = m.group(1).strip()

    val_str = ", ".join(val) if isinstance(val, list) else (str(val) if val is not None else "DELETED")
    new_row = f"| {ts} | {val_str} | {src} |"
    if not history_block:
        history_block = "| Date | Value | Source |\n|------|-------|--------|\n" + new_row
    else:
        history_block += f"\n{new_row}"

    if edges:
        for e in edges:
            if e.get("from") == entity:
                line = f"- {e.get('type')} -> [[{e.get('to')}]]"
                if line not in existing_edges:
                    existing_edges.append(line)
            elif e.get("to") == entity:
                line = f"- {e.get('type')} <- [[{e.get('from')}]]"
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
    with open(page_path(d, entity), "w") as f:
        f.write(content)


def parse_links(content: str) -> List[str]:
    return re.findall(r"\[\[([^\]\|]+)\]\]", content)
