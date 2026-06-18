#!/usr/bin/env bash
# Turnkey setup for the matched AutoDream-vs-OmniMemory MeME comparison on a fresh machine.
# Run from MEME-public/code:  bash bootstrap.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "==> 1/4  Python venv + deps"
python3 -m venv .venvs/baseline_env
# shellcheck disable=SC1091
. .venvs/baseline_env/bin/activate
pip install -q --upgrade pip
pip install -q openai anthropic httpx huggingface_hub tiktoken numpy

echo "==> 2/4  MeME filler32k dataset (100 episodes: 50 pl + 50 sw)"
if [ ! -d data/filler32k_pl ] || [ ! -d data/filler32k_sw ]; then
  python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('meme-benchmark/MEME','meme_filler32k.json',repo_type='dataset',local_dir='../../dataset')"
  python dataset_tools/unpack_dataset.py --input ../../dataset/meme_filler32k.json --output data
else
  echo "    dataset already unpacked, skipping"
fi
echo "    pl episodes: $(ls data/filler32k_pl/episode_*.json 2>/dev/null | wc -l)  sw episodes: $(ls data/filler32k_sw/episode_*.json 2>/dev/null | wc -l)"

echo "==> 3/4  Ollama embedding model (omni_memory vector index uses nomic-embed-text)"
if command -v ollama >/dev/null 2>&1; then
  ollama pull nomic-embed-text
else
  echo "    !! ollama not found. Install from https://ollama.com and run 'ollama pull nomic-embed-text'."
  echo "       (Only omni_memory needs it — for its vector index. The construction LLM is Sonnet via the claude CLI.)"
fi

echo "==> 4/4  Preflight: claude CLI must be installed and logged in"
if command -v claude >/dev/null 2>&1; then
  echo "    claude CLI found: $(command -v claude)"
else
  echo "    !! claude CLI not found. Install Claude Code and 'claude login' (Sonnet is the construction + answer model)."
fi

cat <<'EOF'

Bootstrap complete. Before running the comparison, confirm:
  - `claude` CLI logged in (answers + construction both use claude-code/sonnet)
  - `ollama serve` running and `nomic-embed-text` pulled (omni_memory embeddings)
Then:  bash run_compare.sh
EOF
