# Agent Memory

This project is to evaluate latest trend of different approaches to manage the agent memory for long term interactions. The eval method is based on [MEME-public](https://github.com/SeokwonJung-Jay/MEME-public) with modification to run on Claude code with Claude subscription instead of API key.

## Approaches evaluated
- Karpathy LLM Wiki: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- Claude Auto Memory: https://code.claude.com/docs/en/memory#auto-memory
- Evo-Memory: https://arxiv.org/pdf/2511.20857
- A-Mem: https://arxiv.org/pdf/2502.12110

## How to run eval

Ask claude code:
Run eval for A-Mem/Evo-Memory/Auto Memory/Karpathy LLM Wiki. Pay attend to claude session limits, after each episode, check the session limits, pace the execution to avoid hitting over 90% of claude session limits.

## Eval Results

**Comparison across all 4 approaches (after phase):**

| Task        | In-context | Auto-memory | LLM Wiki  | EvoMemory | A-Mem     |
| ----------- | ---------- | ----------- | --------- | --------- | --------- |
| ER          | 5.0%       | 86.0%       | 3.0%      | 76.0%     | **59.0%** |
| Agg         | 5.0%       | 41.0%       | 65.0%     | 61.0%     | **82.0%** |
| Tr          | 6.0%       | 14.0%       | 64.0%     | 30.0%     | **69.0%** |
| Del         | 34.0%      | 54.0%       | 60.0%     | **97.0%** | 32.0%     |
| Cas         | 1.8%       | 37.8%       | 22.0%     | **56.1%** | 40.2%     |
| Abs         | 63.8%      | 29.2%       | 32.0%     | **62.3%** | 12.3%     |
| **Overall** | **19.6%**  | **42.5%**   | **42.2%** | **52.2%** | **46.7%** |

**A-Mem key observations:**
- Dominant on **Agg (82%)** and **Tr (69%)** — the Zettelkasten link network + context evolution creates excellent cross-entity recall and revision tracking
- Solid **Cas (40%)** — linked notes propagate related facts well
- Weak **Del (32%)** and **Abs (12%)** — atomic notes don't naturally surface deletions; Refine (EvoMemory) is far better for tracking what was removed
- Moderate **ER (59%)** — reformulated summaries lose some verbatim precision vs Auto-memory (86%)

EvoMemory leads overall (52.2%), but A-Mem is the standout for tasks requiring synthesis and history. 

## Cost Analysis
  
  Summary Table (per 100 episodes, filler32k)

  ┌─────────────┬──────────────┬────────────────┬────────────┬─────────┬─────────┬───────────────┐
  │  Approach   │ LLM Calls/ep │ Est. Tokens/ep │ $/ep (API) │ $/100ep │ Wall/ep │ Score (after) │
  ├─────────────┼──────────────┼────────────────┼────────────┼─────────┼─────────┼───────────────┤
  │ In-context  │ 12           │ 503k           │ $1.52      │ $152    │ ~90s    │ 19.6%         │
  ├─────────────┼──────────────┼────────────────┼────────────┼─────────┼─────────┼───────────────┤
  │ Auto-memory │ 17           │ 22.5k          │ $0.11      │ $11     │ 162s    │ 42.5%         │
  ├─────────────┼──────────────┼────────────────┼────────────┼─────────┼─────────┼───────────────┤
  │ LLM Wiki    │ 29           │ 27.5k          │ $0.14      │ $14     │ 215s    │ 42.2%         │
  ├─────────────┼──────────────┼────────────────┼────────────┼─────────┼─────────┼───────────────┤
  │ EvoMemory   │ 19           │ 24.1k          │ $0.16      │ $16     │ 104s    │ 52.2%         │
  ├─────────────┼──────────────┼────────────────┼────────────┼─────────┼─────────┼───────────────┤
  │ A-Mem       │ 22           │ 35.1k          │ $0.17      │ $17     │ 219s    │ 46.7%         │
  └─────────────┴──────────────┴────────────────┴────────────┴─────────┴─────────┴───────────────┘

  Pricing: Claude Sonnet 4.6 ($3/M input, $15/M output). Actual runs used claude -p (Claude Pro subscription).

## LICENSE & DISCLAIMER

[DISCLAIMER](DISCLAIMER.md)
[LICENSE](LICENSE)
