#!/usr/bin/env bash
# Matched AutoDream-vs-OmniMemory comparison, full filler32k (100 episodes), Sonnet everywhere.
# Run from MEME-public/code AFTER bootstrap.sh:  bash run_compare.sh
#
# Held fixed for both agents (only the memory architecture differs):
#   - harness      : run_agent (single-strategy, shared answer prompt) + judge.py
#   - answer model : --model claude-code/sonnet
#   - construction : Sonnet --
#       * auto_memory_dreaming constructs via --model (claude CLI -> Sonnet)
#       * omni_memory constructs via OMNI_CONSTRUCTION_MODEL=claude-code/sonnet
#   - episodes     : full 50 pl + 50 sw
# Resumable: --skip-existing on both run_agent and judge; safe to re-run after a session-limit
# interruption. On hitting a limit, it backs off 30 min and resumes.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venvs/baseline_env/bin/python
MODEL=claude-code/sonnet
export OMNI_CONSTRUCTION_MODEL=claude-code/sonnet   # omni_memory: construct on Sonnet
TOTAL=$(( $(ls data/filler32k_pl/episode_*.json | wc -l) + $(ls data/filler32k_sw/episode_*.json | wc -l) ))

for AT in auto_memory_dreaming omni; do
  echo "================  $AT  ================"
  while :; do
    for D in pl sw; do
      $PY -m eval.run_agent -d "data/filler32k_$D" -o "output/$AT" \
          --agent-type "$AT" --model "$MODEL" -w 1 --skip-existing
    done
    $PY -m eval.judge -d "output/$AT" -o "output/$AT/judge" \
        --judge-model "$MODEL" -w 1 --check-workers 2 --skip-existing
    nj=$(ls "output/$AT/judge/"eval_*.json 2>/dev/null | wc -l)
    echo "[$AT] judged $nj/$TOTAL"
    [ "$nj" -ge "$TOTAL" ] && break
    echo "[$AT] incomplete (likely session limit) — backing off 30 min, then resuming…"
    sleep 1800
  done
done

echo "================  comparison table  ================"
$PY compare.py
