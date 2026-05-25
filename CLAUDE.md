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
│   ├── code/agents/        # 6 built-in memory system implementations
│   ├── code/eval/          # Evaluation runners (run_agent.py, in_context_baseline.py, judge.py)
│   ├── code/dataset_tools/ # Dataset download/unpack utilities
│   ├── code/scripts/       # Bash orchestration scripts
│   └── code/data/          # Unpacked episode files (populated after setup)
└── output/                 # Evaluation results (one subdir per approach)
```

## Setup

```bash
cd MEME-public/code

# 1. Create venv (Python 3.9 suffices for in-context baseline; 3.12+ needed for Karpathy agent)
python3 -m venv .venvs/baseline_env
source .venvs/baseline_env/bin/activate
pip install openai anthropic python-dotenv huggingface_hub tiktoken

# 2. Configure API keys
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

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
source .env

# Run both domains
for DOMAIN in pl sw; do
  python -m eval.in_context_baseline \
    -d data/filler32k_${DOMAIN} \
    -o ../../output/in_context/claude-sonnet-4-6 \
    --model claude-sonnet-4-6 -w 4 --skip-existing
done

# Judge results
python -m eval.judge \
  -d ../../output/in_context/claude-sonnet-4-6 \
  -o ../../output/in_context/claude-sonnet-4-6/judge
```

## External Documents

All files in `external_docs/` are immutable reference documents. Never modify them.

| File | Purpose |
|------|---------|
| `karpathy_llm_wiki.md` | LLM Wiki architecture and workflows |
| `A-Mem.pdf` | A-Mem method paper |
| `Evo_Memory.pdf` | EvoMemory method paper |
| `Meme_memory_eval.pdf` | MeME evaluation framework paper |
