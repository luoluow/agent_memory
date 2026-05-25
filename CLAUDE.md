# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Evaluate 5 different agentic memory management approaches for Claude Code and measure their effectiveness using the MeME evaluation framework (`external_docs/Meme_memory_eval.pdf`).

## The 5 Memory Approaches

| # | Name | Description | Reference |
|---|------|-------------|-----------|
| 1 | **In-context (baseline)** | No external memory; full episode transcript fed directly to the answering LLM | — |
| 2 | **Auto-memory files** | Claude Code's built-in file-based memory system (`.claude/projects/` markdown files) | — |
| 3 | **LLM Wiki** | Incrementally builds and maintains a persistent interlinked wiki of markdown files | `external_docs/karpathy_llm_wiki.md` |
| 4 | **A-Mem** | A-Mem method | `external_docs/A-Mem.pdf` |
| 5 | **EvoMemory** | EvoMemory method | `external_docs/Evo_Memory.pdf` |

## Evaluation Framework

**MeME** (Multi-Entity and Evolving Memory Evaluation) — measures 6 task types across 2 domains (Personal Life, Software Project):

| Task | Measures |
|------|---------|
| ER (Exact Recall) | Verbatim fact reproduction |
| Agg (Aggregation) | Combining scattered facts |
| Tr (Tracking) | Revision history reconstruction |
| Del (Deletion) | Recognizing explicit removals |
| Cas (Cascade) | Update propagation through dependency rules |
| Abs (Absence) | Recognizing uncertainty when no replacement applies |

Dataset configs: `filler32k` (default, 100 episodes), `filler128k` (stress test, 40 episodes), `nofiller` (evidence-only, 100 episodes).

## Repository Layout

```
agent_memory/
├── external_docs/          # Immutable reference papers — do NOT modify
├── MEME-public/            # Forked MeME eval framework (cloned from GitHub)
│   ├── code/agents/        # Memory system implementations
│   │   └── claude_code_adapter.py  # Routes calls through `claude -p` CLI (no API key needed)
│   ├── code/eval/          # Evaluation runners (run_agent.py, in_context_baseline.py, judge.py)
│   ├── code/eval_docs/     # Notes and exported results per evaluation run
│   ├── code/dataset_tools/ # Dataset download/unpack utilities
│   ├── code/scripts/       # Bash orchestration scripts
│   └── code/data/          # Unpacked episode files (populated after setup)
└── output/                 # Evaluation results (one subdir per approach)
    └── in_context/
        └── claude-code/    # In-context baseline results (100 episodes, filler32k)
            └── judge/      # Per-episode judge outputs + aggregated scores
```

## Setup

```bash
cd MEME-public/code

# 1. Create venv (Python 3.9 suffices for in-context baseline; 3.12+ needed for Karpathy agent)
python3 -m venv .venvs/baseline_env
source .venvs/baseline_env/bin/activate
pip install openai anthropic python-dotenv huggingface_hub tiktoken

# 2. API keys
# For claude-code model (routes through `claude -p` CLI — uses your Claude Pro subscription):
#   No API key needed. Requires `claude` CLI to be installed and logged in.
# For claude-sonnet-4-6 or other Anthropic models via API:
#   export ANTHROPIC_API_KEY=sk-...
# For judge with OpenAI models (gpt-4o default):
#   export OPENAI_API_KEY=sk-...

# 3. Download and unpack dataset
python3 -c "
from huggingface_hub import hf_hub_download
for name in ['meme_filler32k.json']:
    hf_hub_download('meme-benchmark/MEME', name, repo_type='dataset', local_dir='../../dataset')
"
python3 dataset_tools/unpack_dataset.py --input ../../dataset/meme_filler32k.json --output data
```

## Running Evaluations

### Baseline (in-context, no memory)
```bash
cd MEME-public/code
source .venvs/baseline_env/bin/activate

# Run both domains via Claude CLI (no API key needed)
for DOMAIN in pl sw; do
  python -m eval.in_context_baseline \
    -d data/filler32k_${DOMAIN} \
    -o ../../output/in_context/claude-code \
    --model claude-code -w 1 --skip-existing
done

# Judge results (also via Claude CLI)
python -m eval.judge \
  -d ../../output/in_context/claude-code \
  -o ../../output/in_context/claude-code/judge \
  --judge-model claude-code \
  -w 1 --check-workers 4 --skip-existing
```

**Note:** Use `-w 1` for `claude-code` model to avoid overwhelming the CLI with concurrent subprocesses. The judge's `--skip-existing` flag was added in our fork to support resuming interrupted runs.

### Completed runs

| Approach | Model | Dataset | Agent output | Judge output |
|----------|-------|---------|--------------|--------------|
| In-context baseline | claude-code | filler32k (100 ep) | `output/in_context/claude-code/` | `output/in_context/claude-code/judge/` |

#### In-context baseline results (filler32k, 100 episodes)

| Task | Phase | Pass | Total | % | Notes |
|------|-------|------|-------|---|-------|
| ER | before | 0 | 100 | 0.0% | |
| ER | after | 5 | 100 | 5.0% | |
| Agg | after | 5 | 100 | 5.0% | |
| Tr | after | 6 | 100 | 6.0% | |
| Del | before | 5 | 100 | 5.0% | |
| Del | after | 34 | 100 | 34.0% | real=2, trivial=32 |
| Cas | before | 11 | 164 | 6.7% | |
| Cas | after | 3 | 164 | 1.8% | real=3, trivial=0 |
| Abs | before | 9 | 130 | 6.9% | |
| Abs | after | 83 | 130 | 63.8% | real=0, trivial=83 |
| **Overall** | **before** | **25** | **494** | **5.1%** | |
| **Overall** | **after** | **136** | **694** | **19.6%** | |

## External Documents

All files in `external_docs/` are immutable reference documents. Never modify them.

| File | Purpose |
|------|---------|
| `karpathy_llm_wiki.md` | LLM Wiki architecture and workflows |
| `A-Mem.pdf` | A-Mem method paper |
| `Evo_Memory.pdf` | EvoMemory method paper |
| `Meme_memory_eval.pdf` | MeME evaluation framework paper |
