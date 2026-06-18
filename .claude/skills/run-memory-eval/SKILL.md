---
name: run-memory-eval
description: Run the matched AutoDream-vs-OmniMemory MeME comparison (full 100 episodes, Sonnet construction + answer). Use when the user wants to reproduce or kick off this memory-systems eval, check its progress, or print the per-task results table.
---

# Run the matched AutoDream-vs-OmniMemory eval

Reproduces the AutoDream-vs-OmniMemory comparison on the MeME benchmark with the earlier
study's confounds removed: **same harness, same answer model, same construction model, same
100 episodes** — only the memory architecture varies.

| Held fixed | Value |
|---|---|
| Harness | MeME `run_agent` (single-strategy, shared prompt) + `judge.py` |
| Answer model | `claude-code/sonnet` |
| Construction model | `claude-code/sonnet` for **both** agents |
| Episodes | full `filler32k` (50 `pl` + 50 `sw`) |

- **AutoDream** = `auto_memory_dreaming` (auto-memory ingest + transcript-aware dream).
- **OmniMemory** = `omni` (decoupled per-purpose indices; vector index uses Ollama
  `nomic-embed-text` for embeddings only — the construction LLM is Sonnet).

This eval is **entirely self-contained in the `agent_memory` repo** — it does NOT need the
separate omniservice / `omni_memory` repo, wherever that is on disk.

## Where everything lives

All commands run from `<agent_memory>/MEME-public/code`. Find that dir relative to this skill:
it's `../../MEME-public/code` from the repo root (`.claude/skills/run-memory-eval/`). If unsure,
locate it with `git rev-parse --show-toplevel` then `cd "$(git rev-parse --show-toplevel)/MEME-public/code"`.

Kit files (already committed):
- `bootstrap.sh` — one-time setup
- `run_compare.sh` — the run (resumable)
- `compare.py` — prints the comparison table

## Steps

### 1. Preflight (report any failures to the user, don't silently continue)
- `claude` CLI installed and logged in — provides Sonnet for construction **and** answering.
- `ollama serve` running with `nomic-embed-text` pulled — OmniMemory's vector index.
- The venv exists at `MEME-public/code/.venvs/baseline_env` (created by `bootstrap.sh`).

### 2. One-time setup (skip if `data/filler32k_pl` already exists)
```bash
bash bootstrap.sh
```
Creates the venv + deps, downloads/unpacks `filler32k` (100 episodes), pulls `nomic-embed-text`.

### 3. Run the comparison
```bash
bash run_compare.sh
```
Runs **both** agents at full 100 with matched Sonnet construction + answer, then judges, then
prints the table. It is **resumable** (`--skip-existing` on both `run_agent` and `judge`) and
**backs off 30 min on a Claude session limit**, then continues.

**Run it in the background** (this is a long, session-limited job — do not block on it):
launch with `run_in_background`, then poll progress with the command in step 4 rather than
waiting. Tell the user it's running and roughly how to check on it. Never use `pkill -f` to
stop anything (it matches the controlling shell — exit 144); kill by specific PID if needed.

### 4. Check progress / reprint results (anytime)
```bash
# how many episodes each agent has been judged on (out of 100):
ls output/auto_memory_dreaming/judge/eval_*.json 2>/dev/null | wc -l
ls output/omni/judge/eval_*.json 2>/dev/null | wc -l
# the per-task comparison table:
.venvs/baseline_env/bin/python compare.py
```

## Outputs
```
MEME-public/code/output/auto_memory_dreaming/{agent_*.json, judge/eval_*.json}
MEME-public/code/output/omni/{agent_*.json, judge/eval_*.json}
```
`compare.py` aggregates the after-phase `u_pass` per task (ER / Agg / Tr / Del / Cas / Abs) into
a two-column AutoDream-vs-OmniMemory table.

## Knobs (mention only if the user asks for variants)
- **Cheaper construction** (cost-tier story, off Sonnet): set `OMNI_CONSTRUCTION_MODEL` to
  `deepseek-chat` (needs `DEEPSEEK_API_KEY`) or a local Ollama model (`gemma4-ctx32k`) for
  OmniMemory; pass the matching `--model` for AutoDream. Keep the two matched.
- **Smoke run**: point `run_agent -d` at a dir with only a few `episode_*.json` files.

Full background: `REPRODUCE.md` (repo root) and `omniservice/docs/autodream_comparison.md`.
