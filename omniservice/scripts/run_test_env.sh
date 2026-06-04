#!/usr/bin/env bash
# Run the OmniService test environment:
#   - boots the FastAPI service (if not already running)
#   - runs the full pytest suite, including LIVE end-to-end tests
#   - live tests use a dedicated client id ("test") so real client data is untouched
#
# Usage:  scripts/run_test_env.sh
# Env:    OMNI_CLIENT_ID (default "test"), OMNI_PORT (default 11435),
#         OMNI_STORAGE_ROOT (default ~/.omni)
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python

export OMNI_CLIENT_ID="${OMNI_CLIENT_ID:-test}"
export OMNI_PORT="${OMNI_PORT:-11435}"
export OMNI_URL="${OMNI_URL:-http://127.0.0.1:${OMNI_PORT}}"

if [ "$OMNI_CLIENT_ID" = "claude-code" ]; then
  echo "refusing to run tests under the default client id 'claude-code'." >&2
  exit 1
fi

# Warn if Ollama is unreachable (live tests need it).
if ! curl -s http://localhost:11434/api/tags -m 2 >/dev/null 2>&1; then
  echo "WARNING: Ollama not reachable at :11434 — live tests will fail." >&2
fi

STARTED=""
if ! curl -s "$OMNI_URL/health" -m 1 >/dev/null 2>&1; then
  echo "starting OmniService on $OMNI_URL ..."
  $PY -m omni.server > /tmp/omni_test_server.log 2>&1 &
  SRV=$!
  STARTED=1
  trap '[ -n "$STARTED" ] && kill "$SRV" 2>/dev/null || true' EXIT
  for _ in $(seq 1 30); do
    curl -s "$OMNI_URL/health" -m 1 >/dev/null 2>&1 && break
    sleep 0.5
  done
fi

echo "running test suite (client_id=$OMNI_CLIENT_ID) ..."
OMNI_LIVE=1 $PY -m pytest tests/ -v "$@"
