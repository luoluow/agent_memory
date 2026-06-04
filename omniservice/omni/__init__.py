"""OmniService — an independent, namespaced memory service.

OmniMemory packaged as a standalone FastAPI service with a local-LLM (Ollama)
memory pipeline. Clients (Claude Code via hooks + MCP, or any agent) push raw
interactions to /ingest and fetch context-scoped memory from /retrieve.

See external_docs/OmniMemory_design.md → "Claude Code Integration".
"""

__version__ = "0.1.0"
