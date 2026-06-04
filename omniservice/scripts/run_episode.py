#!/usr/bin/env python
"""Run one MeME episode through OmniService end-to-end.

Ingests an episode's sessions in order via the HTTP API (each session -> /ingest,
which archives + queues EXTRACT), forces a VERIFY pass, then issues the episode's
questions as /retrieve queries and reports a recall-in-context proxy (does the gold
answer appear in the retrieved context?).

Note: this measures *retrieval recall*, not final answer accuracy — answering is the
client LLM's job in the real system / the MeME eval harness. It's a fast signal that
OmniService retained and surfaced the right facts.

Usage:
  .venv/bin/python scripts/run_episode.py \
      ../MEME-public/code/data/filler32k_pl/episode_001.json [--client meme] [--phase after]
"""
import argparse
import json
import os
import time

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", help="path to episode_NNN.json")
    ap.add_argument("--client", default="meme")
    ap.add_argument("--ns", default=None, help="namespace (default: episode_<id>_<domain>)")
    ap.add_argument("--url", default=os.environ.get("OMNI_URL", "http://127.0.0.1:11435"))
    ap.add_argument("--phase", choices=["before", "after", "all"], default="after")
    args = ap.parse_args()

    ep = json.load(open(args.episode))
    ns = args.ns or f"episode_{ep['episode_id']:03d}_{ep['domain']}"
    B = args.url
    evid = set(ep["evidence_session_indices"])
    sessions = ep["sessions"]
    before_pos = ep["before_questions"]["position_after_session"]
    after_pos = ep["after_questions"]["position_after_session"]
    end = {"before": before_pos + 1, "after": after_pos + 1, "all": len(sessions)}[args.phase]

    print(f"episode {ep['episode_id']} ({ep['domain']}): {len(sessions)} sessions "
          f"({len(evid)} evidence at {sorted(evid)}); ingesting first {end}", flush=True)

    httpx.request("DELETE", f"{B}/namespace",
                  params={"client_id": args.client, "namespace": ns}, timeout=120)

    t0 = time.time()
    for i, s in enumerate(sessions[:end]):
        turns = [{"role": t["role"], "content": t["content"]} for t in s.get("conversation", [])]
        if not turns:
            continue
        httpx.post(f"{B}/ingest", json={
            "client_id": args.client, "namespace": ns, "turns": turns,
            "timestamp": s.get("timestamp", ""), "source": f"session_{i}",
        }, timeout=120).raise_for_status()
        tag = "EVID" if (i in evid or s.get("type") == "evidence") else "fill"
        print(f"  [{i:2d}] {tag} {s.get('timestamp',''):22} {len(turns):2d} turns -> queued", flush=True)

    print(f"\ningested {end} sessions in {time.time()-t0:.0f}s; forcing VERIFY "
          f"(waits for all EXTRACT to finish)...", flush=True)
    tv = time.time()
    v = httpx.post(f"{B}/verify", json={"client_id": args.client, "namespace": ns}, timeout=3600).json()
    print(f"VERIFY ({time.time()-tv:.0f}s): {v}", flush=True)

    qkey = "before_questions" if args.phase == "before" else "after_questions"
    qs = ep[qkey]["questions"]
    print(f"\n=== recall-in-context on {len(qs)} {qkey} ===", flush=True)
    hits = 0
    by_task = {}
    for q in qs:
        ctx = httpx.post(f"{B}/retrieve", json={
            "client_id": args.client, "namespace": ns, "query": q["question"], "mode": "search",
        }, timeout=120).json()["context"]
        gold = (q.get("gold_answer") or "").strip()
        vals = [v.strip() for v in gold.split(",") if v.strip()] or [gold]
        present = bool(gold) and all(v.lower() in ctx.lower() for v in vals)
        hits += present
        tt = q["task_type"]
        d = by_task.setdefault(tt, [0, 0]); d[1] += 1; d[0] += present
        print(f"  [{tt:3}] {'OK  ' if present else 'MISS'} gold={gold!r}", flush=True)

    print(f"\nrecall-in-context: {hits}/{len(qs)}  "
          + "  ".join(f"{t}:{c[0]}/{c[1]}" for t, c in sorted(by_task.items())), flush=True)
    print(f"namespace '{ns}' — inspect in the UI: {B}/ui/", flush=True)


if __name__ == "__main__":
    main()
