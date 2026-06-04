# OmniService

OmniMemory packaged as a **standalone, namespaced memory service**. Clients (Claude
Code via hooks + MCP, or any agent) push raw interactions to `/ingest` and fetch
context-scoped memory from `/retrieve`. All memory work (EXTRACT / RELATE / VERIFY /
COMPRESS) runs on a local LLM via Ollama — off the client's quota, in the background.

See `../external_docs/OmniMemory_design.md` → **Claude Code Integration** for the design.

```
   Claude Code ──hooks──▶ omni CLI ─┐
               ──MCP────▶ omni mcp ─┤ HTTP :11435
                                    ▼
                            omni-memoryd (FastAPI)
                              OmniEngine ──▶ Ollama (gemma4)
                                    ▼
                            ~/.omni/<namespace>/
```

## Layout

| File | Role |
|------|------|
| `omni/engine.py` | OmniEngine: namespaced ingest → background EXTRACT → debounced VERIFY → ranked retrieve |
| `omni/storage.py` | Per-namespace on-disk store (raw / state / history / deletions / pages / cursor) |
| `omni/server.py` | FastAPI HTTP service |
| `omni/cli.py` | `omni` CLI — invoked by Claude Code hooks (transcript parsing + cursor dedup) |
| `omni/mcp_server.py` | MCP stdio server — `memory_search` / `memory_note` tools |
| `omni/llm.py`, `omni/prompts.py`, `omni/config.py` | Ollama client, prompts, env config |

## Setup

```bash
cd omniservice
# venv is self-contained in ./.venv (Python 3.11)
.venv/bin/python -m pip install -e .        # installs the `omni` entry point

# Local model (Ollama must be running) — serves both EXTRACT and VERIFY:
ollama pull gemma4:e4b
```

## Run

```bash
.venv/bin/omni serve         # starts FastAPI on 127.0.0.1:11435
# or: .venv/bin/python -m omni.server
```

## Web UI — memory inspector

Open **http://127.0.0.1:11435/** (redirects to `/ui/`). Pick a client + namespace to:
- **Actions** tab — chronological audit log of every EXTRACT / RELATE / VERIFY call:
  op, timestamp, model, duration, and applied effect. Expand any row for the full
  **LLM request** (system prompt, user message, model + sampling params) and
  **response** (raw output, parsed JSON), plus the applied memory writes
  (errors flagged red).
- **Memory snapshot** tab — the current assembled memory for that namespace.

Backed by `GET /actions?client_id=…&namespace=…` over the per-namespace `actions.jsonl`
audit log (written by the engine on every memory-op LLM call).

Quick check:
```bash
curl -s localhost:11435/health
curl -s -X POST localhost:11435/ingest  -H 'Content-Type: application/json' \
  -d '{"client_id":"alice","namespace":"/proj/x","turns":[{"role":"user","content":"My medication is Quelmithin."}],"timestamp":"2023/03/17"}'
curl -s -X POST localhost:11435/retrieve -H 'Content-Type: application/json' \
  -d '{"client_id":"alice","namespace":"/proj/x","query":"medication","mode":"search"}'
```

`client_id` is **required** on every request and partitions storage as
`~/.omni/<client_id>/<namespace>/`. Test runs use a separate client id so they never
touch real data. Wipe a whole client with `DELETE /client?client_id=...`.

## Configuration (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `OMNI_STORAGE_ROOT` | `~/.omni` | Storage root (`<client_id>/<namespace>/`) |
| `OMNI_CLIENT_ID` | `claude-code` | Default client id for the CLI/MCP clients |
| `OMNI_HOST` / `OMNI_PORT` | `127.0.0.1` / `11435` | HTTP bind |
| `OMNI_OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Local LLM endpoint |
| `OMNI_EXTRACT_MODEL` | `gemma4:e4b` | EXTRACT / RELATE / COMPRESS |
| `OMNI_VERIFY_MODEL` | `gemma4:e4b` | VERIFY |
| `OMNI_VERIFY_DEBOUNCE_SECONDS` | `20` | Idle window before background VERIFY |
| `OMNI_URL` | (host/port) | Override the URL CLI/MCP clients talk to |
| `OMNI_NAMESPACE` | (cwd) | Force namespace for the MCP server |

## Claude Code integration

The `omni` binary must be on `PATH` (e.g. symlink `.venv/bin/omni`), and the service
must be running (`omni serve`).

**`~/.claude/settings.json`** — deterministic push + baseline retrieve seed:
```jsonc
{
  "hooks": {
    "SessionStart": [{ "hooks": [{ "type": "command", "command": "omni retrieve --session-start" }]}],
    "Stop":         [{ "hooks": [{ "type": "command", "command": "omni ingest" }]}]
  }
}
```
The hooks read the hook payload (`transcript_path`, `cwd`) from stdin; namespace
defaults to the project `cwd`. `omni ingest` parses the transcript JSONL, sends only
turns newer than its per-transcript cursor, and the SessionStart hook injects the
memory seed as `additionalContext`.

**`.mcp.json`** (or `claude mcp add`) — context-aware pull:
```jsonc
{ "mcpServers": { "omni": { "command": "omni", "args": ["mcp"] } } }
```
Exposes `memory_search(query)` and `memory_note(text)`.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q     # mocked engine/storage logic (no Ollama)
scripts/run_test_env.sh                  # full suite incl. LIVE e2e + MCP tools
```
`tests/test_smoke.py` runs with Ollama mocked. `tests/test_live.py` is gated by
`OMNI_LIVE=1` and exercises the real model and MCP tools; `run_test_env.sh` boots the
service and runs everything under client id `test` (wiped before/after, refuses to run
as `claude-code`). A live run requires Ollama with `gemma4:e4b` pulled.
