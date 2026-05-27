# Agent Memory

This project is to evaluate latest trend of different approaches to manage the agent memory for long term interactions. The eval method is based on [MEME-public](https://github.com/SeokwonJung-Jay/MEME-public) with modification to run on Claude code with Claude subscription instead of API key.

## Approaches evaluted
- Karpathy LLM Wiki: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- Claude Auto Memory: https://code.claude.com/docs/en/memory#auto-memory
- Evo-Memory: https://arxiv.org/pdf/2511.20857
- A-Mem: https://arxiv.org/pdf/2502.12110

## How to run eval

Ask claude code:
Run eval for A-Mem/Evo-Memory/Auto Memory/Karpathy LLM Wiki. Pay attend to claude session limits, after each episode, check the session limits, pace the execution to avoid hitting over 90% of claude session limits.

## LICENSE & DISCLAIMER

[DISCLAIMER.md](DISCLAIMER.md)
[LICENSE.md](LICENSE.md)
