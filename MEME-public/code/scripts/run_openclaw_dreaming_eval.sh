#!/usr/bin/env bash
# OpenClaw memory-core + dreaming consolidation eval on MeME filler32k (pl + sw), then judge.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venvs/baseline_env/bin/activate
OUT=../../output/openclaw_dreaming/claude-code
mkdir -p "$OUT"
for DOMAIN in pl sw; do
  echo "=== INGEST+ANSWER: $DOMAIN ==="
  python -m eval.run_agent -d "data/filler32k_${DOMAIN}" -o "$OUT" \
    --agent-type openclaw_dreaming --model claude-code -w 1 --skip-existing
done
echo "=== JUDGE ==="
python -m eval.judge -d "$OUT" -o "$OUT/judge" \
  --judge-model claude-code -w 1 --check-workers 4 --skip-existing
echo "=== DONE ==="
