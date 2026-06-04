"""FastAPI HTTP service for OmniService.

Endpoints:
    POST /ingest      push raw interaction turns; archive + background EXTRACT
    POST /retrieve    fetch context-scoped memory (no LLM call)
    POST /verify      force a synchronous VERIFY pass (flush)
    GET  /snapshot    full assembled memory for a namespace
    GET  /namespaces  list known namespaces
    DELETE /namespace reset a namespace
    GET  /health      liveness
"""

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from omni import __version__, config, storage
from omni.engine import OmniEngine

app = FastAPI(title="OmniService", version=__version__)
engine = OmniEngine()

_WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    role: str
    content: str


class IngestRequest(BaseModel):
    client_id: str
    namespace: str
    turns: List[Turn]
    timestamp: str = ""
    source: str = ""


class RetrieveRequest(BaseModel):
    client_id: str
    namespace: str
    query: str = ""
    mode: str = Field(default="search", description="search | session-start | full")


class NamespaceQuery(BaseModel):
    client_id: str
    namespace: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": __version__,
            "storage_root": str(config.STORAGE_ROOT),
            "extract_model": engine.extract_model, "verify_model": engine.verify_model}


@app.post("/ingest")
def ingest(req: IngestRequest):
    if not req.turns:
        raise HTTPException(status_code=400, detail="no turns provided")
    turns = [t.model_dump() for t in req.turns]
    result = engine.ingest(req.client_id, req.namespace, turns,
                           timestamp=req.timestamp, source=req.source)
    return {"ok": True, **result}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    context = engine.retrieve(req.client_id, req.namespace, query=req.query, mode=req.mode)
    return {"client_id": req.client_id, "namespace": req.namespace,
            "mode": req.mode, "context": context}


@app.post("/verify")
def verify(req: NamespaceQuery):
    return {"client_id": req.client_id, "namespace": req.namespace,
            **engine.verify_now(req.client_id, req.namespace)}


@app.get("/snapshot")
def snapshot(client_id: str, namespace: str):
    return {"client_id": client_id, "namespace": namespace,
            "text": engine.snapshot(client_id, namespace)}


@app.get("/actions")
def actions(client_id: str, namespace: str, limit: int = 200):
    d = storage.ns_dir(client_id, namespace)
    items = storage.read_actions(d, limit=limit)
    items.reverse()  # newest first
    return {"client_id": client_id, "namespace": namespace, "count": len(items),
            "actions": items}


@app.get("/namespaces")
def namespaces(client_id: str):
    return {"client_id": client_id, "namespaces": storage.list_namespaces(client_id)}


@app.get("/clients")
def clients():
    return {"clients": storage.list_clients()}


@app.delete("/namespace")
def delete_namespace(client_id: str, namespace: str):
    engine.reset(client_id, namespace)
    return {"ok": True, "client_id": client_id, "namespace": namespace}


@app.delete("/client")
def delete_client(client_id: str):
    return {"ok": True, **engine.reset_client(client_id)}


@app.get("/")
def root():
    return RedirectResponse(url="/ui/")


# Web UI (static, vanilla JS — browses the action log + memory snapshot).
app.mount("/ui", StaticFiles(directory=_WEB_DIR, html=True), name="ui")


def main():
    import uvicorn
    uvicorn.run("omni.server:app", host=config.HOST, port=config.PORT, log_level="info")


if __name__ == "__main__":
    main()
