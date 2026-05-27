#!/usr/bin/env python3
"""
Paced single-episode runner for claude-code agents.

Runs episodes one at a time with inter-episode health checks and sleep to
avoid hitting Claude Pro session limits, especially when Claude is also in
use in other projects simultaneously.

After each episode:
  1. Checks output for session-limit contamination (re-queues if found)
  2. Health-checks Claude with a quick claude -p call
  3. Sleeps --inter-sleep seconds (proactive pacing)
  On any limit detection: sleeps --limit-sleep seconds then retries.

Usage:
  cd MEME-public/code
  source .venvs/baseline_env/bin/activate
  python scripts/run_paced.py --domain pl
  python scripts/run_paced.py --domain both --inter-sleep 90

Default output: ../../output/{agent}/claude-code/
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_DIR.parent.parent

LIMIT_PHRASES = [
    "session limit", "rate limit", "too many requests",
    "usage limit", "resets", "claude.ai/upgrade",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_claude_health(timeout: int = 30) -> str:
    """Quick claude -p call — returns 'ok' | 'limit' | 'unknown'."""
    try:
        result = subprocess.run(
            [
                "claude", "-p",
                "--output-format", "text",
                "--no-session-persistence",
                "--system-prompt", "Reply with exactly the word: OK",
            ],
            input="OK",
            capture_output=True, text=True, timeout=timeout,
            cwd=str(CODE_DIR),
        )
        combined = (result.stdout + result.stderr).lower()
        if any(p in combined for p in LIMIT_PHRASES):
            return "limit"
        if result.returncode != 0:
            return "unknown"
        return "ok"
    except subprocess.TimeoutExpired:
        return "unknown"
    except Exception as e:
        print(f"  [health] error: {e}", flush=True)
        return "unknown"


def is_contaminated(out_path: str) -> bool:
    """Return True if the output file contains session-limit strings in answers."""
    try:
        with open(out_path) as f:
            data = json.load(f)
        for key in ("before_answers", "after_answers"):
            for q in data.get(key, []):
                ans = (q.get("agent_answer") or "").lower()
                if any(p in ans for p in LIMIT_PHRASES):
                    return True
    except Exception:
        pass
    return False


def wait_for_reset(limit_sleep: int) -> None:
    until = time.strftime("%H:%M:%S", time.localtime(time.time() + limit_sleep))
    print(f"  [paced] Sleeping {limit_sleep // 60} min for limit reset (until ~{until})...",
          flush=True)
    time.sleep(limit_sleep)
    print("  [paced] Resuming.", flush=True)


def health_check_with_retry(limit_sleep: int) -> None:
    """Block until health check passes (retrying on limit)."""
    while True:
        health = check_claude_health()
        if health == "ok":
            return
        if health == "limit":
            print("  [paced] Session limit detected in health check.", flush=True)
            wait_for_reset(limit_sleep)
        else:
            # unknown (timeout / error) — proceed anyway
            print("  [paced] Health check uncertain (unknown), proceeding.", flush=True)
            return


def run_episode(ep_path: Path, output_dir: str, agent: str, model: str) -> int:
    """Run one episode via subprocess. Returns returncode."""
    cmd = [
        sys.executable, "-m", "eval.run_agent",
        "-e", str(ep_path),
        "-o", output_dir,
        "--agent-type", agent,
        "--model", model,
        "-w", "1",
    ]
    result = subprocess.run(cmd, cwd=str(CODE_DIR))
    return result.returncode


def out_path_for(output_dir: str, domain: str, ep_id: int,
                 agent: str, model: str) -> str:
    model_tag = model.replace("/", "-")
    return os.path.join(output_dir, f"agent_{domain}_{ep_id:03d}_{agent}_{model_tag}.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paced single-episode runner for claude-code agents",
    )
    parser.add_argument("--domain", default="pl", choices=["pl", "sw", "both"],
                        help="Domain(s) to evaluate (default: pl)")
    parser.add_argument("--agent", default="wiki",
                        choices=["auto_memory", "wiki", "evomem", "amem", "md_file"],
                        help="Agent type (default: wiki)")
    parser.add_argument("--model", default="claude-code",
                        help="Model identifier (default: claude-code)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: ../../output/{agent}/{model}/)")
    parser.add_argument("--inter-sleep", type=int, default=120,
                        help="Seconds to sleep between episodes (default: 120)")
    parser.add_argument("--limit-sleep", type=int, default=1800,
                        help="Seconds to sleep on limit detection (default: 1800 = 30 min)")
    parser.add_argument("--no-health-check", action="store_true",
                        help="Skip health checks (faster, riskier)")
    args = parser.parse_args()

    domains = ["pl", "sw"] if args.domain == "both" else [args.domain]

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        model_tag = args.model.replace("/", "-")
        output_dir = str(REPO_ROOT / "output" / args.agent / model_tag)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[paced] agent={args.agent}  model={args.model}  domain(s)={domains}", flush=True)
    print(f"[paced] output → {output_dir}", flush=True)
    print(f"[paced] inter-sleep={args.inter_sleep}s  limit-sleep={args.limit_sleep}s", flush=True)

    for domain in domains:
        data_dir = CODE_DIR / "data" / f"filler32k_{domain}"
        episodes = sorted(data_dir.glob("episode_*.json"))
        print(f"\n[paced] Domain {domain}: {len(episodes)} episodes found", flush=True)

        for ep_path in episodes:
            m = re.search(r"(\d+)", ep_path.stem)
            if not m:
                continue
            ep_id = int(m.group(1))
            out = out_path_for(output_dir, domain, ep_id, args.agent, args.model)

            # Skip clean existing outputs
            if os.path.exists(out):
                if is_contaminated(out):
                    print(f"[paced] {domain}-{ep_id:03d}: contaminated output, removing.",
                          flush=True)
                    os.remove(out)
                else:
                    print(f"[paced] {domain}-{ep_id:03d}: skip (clean output exists)",
                          flush=True)
                    continue

            # Pre-episode health check
            if not args.no_health_check:
                health_check_with_retry(args.limit_sleep)

            print(f"\n[paced] {domain}-{ep_id:03d}: starting episode...", flush=True)
            t0 = time.time()
            rc = run_episode(ep_path, output_dir, args.agent, args.model)
            elapsed = time.time() - t0

            # Post-episode contamination check
            if os.path.exists(out) and is_contaminated(out):
                print(f"[paced] {domain}-{ep_id:03d}: output contaminated after run. "
                      f"Removing and retrying once.", flush=True)
                os.remove(out)
                wait_for_reset(args.limit_sleep)
                if not args.no_health_check:
                    health_check_with_retry(args.limit_sleep)
                rc = run_episode(ep_path, output_dir, args.agent, args.model)
                elapsed += time.time() - t0

            status = "OK" if (os.path.exists(out) and not is_contaminated(out)) else "WARN"
            print(f"[paced] {domain}-{ep_id:03d}: {status} ({elapsed:.0f}s). "
                  f"Sleeping {args.inter_sleep}s...", flush=True)
            time.sleep(args.inter_sleep)

    print("\n[paced] All done!", flush=True)
    print(f"[paced] Run judge with:", flush=True)
    print(f"  python -m eval.judge -d {output_dir} -o {output_dir}/judge "
          f"--judge-model claude-code -w 1 --check-workers 4 --skip-existing", flush=True)


if __name__ == "__main__":
    main()
