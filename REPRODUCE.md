# Reproduce: AutoDream vs OmniMemory (matched, full 100, Sonnet)

Turnkey kit to re-run the **AutoDream-vs-OmniMemory** MeME comparison cleanly on a fresh
machine, with the three confounds from the earlier study removed: **same harness, same answer
model, same construction model, same 100 episodes**. The only variable is the memory
architecture.

| Held fixed | Value |
|---|---|
| Harness | MeME `run_agent` (single-strategy, shared answer prompt) + `judge.py` |
| Answer model | `claude-code/sonnet` (Sonnet via the `claude` CLI) |
| Construction model | `claude-code/sonnet` for **both** agents |
| Episodes | full `filler32k` — 50 `pl` + 50 `sw` |

- **AutoDream** = `auto_memory_dreaming` — auto-memory ingest + a transcript-aware "dream"
  consolidation. Constructs via `--model` (the claude CLI), so `--model claude-code/sonnet`
  puts its ingest **and** dream on Sonnet.
- **OmniMemory** = `omni` — decoupled per-purpose indices (state / chains / deletions / rules /
  vectors). Constructs via `OMNI_CONSTRUCTION_MODEL` (new knob; routes EXTRACT/VERIFY/RELATE/
  COMPRESS/RECONSTRUCT through the claude CLI). Its **vector index** still uses Ollama
  `nomic-embed-text` (embeddings only — separate from the construction LLM).

## Prerequisites on the new machine
1. **`claude` CLI** installed and logged in (provides Sonnet for both construction + answering).
   - The run is large (100 ep × 2 agents × ~14 Q + ingest/dream/verify, all on Sonnet) and is
     **subject to Claude session limits**. `run_compare.sh` is resumable and backs off 30 min on
     a limit, then continues — leave it running, or re-invoke it any time.
2. **Ollama** running with the embedding model: `ollama serve` + `ollama pull nomic-embed-text`
   (only OmniMemory needs it, for its vector index).
3. Python 3.9+ and `git`.

## Steps
```bash
git clone <this-repo> agent_memory && cd agent_memory/MEME-public/code

bash bootstrap.sh      # venv + deps + downloads/unpacks filler32k + pulls nomic-embed-text
bash run_compare.sh    # runs BOTH agents at full 100, Sonnet construction+answer, then judges
                       # resumable: safe to Ctrl-C and re-run; --skip-existing picks up where it left off
```
`run_compare.sh` finishes by printing the per-task comparison table. To reprint it later
without re-running:
```bash
.venvs/baseline_env/bin/python compare.py
```

## Outputs
```
MEME-public/code/output/auto_memory_dreaming/{agent_*.json, judge/eval_*.json}
MEME-public/code/output/omni/{agent_*.json, judge/eval_*.json}
```
`compare.py` aggregates the after-phase `u_pass` per task from the `judge/` dirs.

## What this isolates (vs the earlier study)
The prior 100-ep study (`MEMORY_SYSTEMS_COMPARISON.md`) put AutoDream at 66% and OmniMemory's
headline at 71.2%, but they differed on answer path, answer model, and harness. This run holds
all three fixed and matches the construction model (Sonnet for both), so any remaining gap is
attributable to the **memory design** — e.g. the prediction that consolidation (AutoDream) loses
**Tracking** while OmniMemory's decoupled `chains.json` keeps it high. See
`omniservice/docs/autodream_comparison.md` for the hypotheses.

## Knobs
- Cheaper construction (architecture-only question, off Sonnet): set
  `OMNI_CONSTRUCTION_MODEL=deepseek-chat` (needs `DEEPSEEK_API_KEY`) or a local Ollama model
  (`gemma4-ctx32k`) for OmniMemory; for AutoDream pass `--model` with the cheaper model. Keep
  them matched.
- Smaller smoke run: temporarily point `run_agent -d` at a directory with a few `episode_*.json`
  files, or edit the domains loop in `run_compare.sh`.
