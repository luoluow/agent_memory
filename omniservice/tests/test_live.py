"""Live end-to-end tests against a running OmniService + real Ollama model.

Gated: only run when OMNI_LIVE=1 and a server is reachable at OMNI_URL. Uses a
dedicated client id (default "test") so it never touches real client data, and
wipes that client before and after.

Run via:  scripts/run_test_env.sh
"""

import os

import httpx
import pytest

LIVE = os.environ.get("OMNI_LIVE") == "1"
CLIENT = os.environ.get("OMNI_CLIENT_ID", "test")
PORT = os.environ.get("OMNI_PORT", "11435")
BASE = os.environ.get("OMNI_URL", f"http://127.0.0.1:{PORT}")
NS = "/test/e2e"

pytestmark = pytest.mark.skipif(not LIVE, reason="set OMNI_LIVE=1 (and run a server) for live tests")


def _wipe():
    try:
        httpx.delete(f"{BASE}/client", params={"client_id": CLIENT}, timeout=30)
    except Exception:
        pass


@pytest.fixture(scope="module", autouse=True)
def clean_client():
    # Guard: refuse to run if the test client id looks like a real one.
    assert CLIENT != "claude-code", "refusing to run live tests against the default client id"
    _wipe()
    yield
    _wipe()


def test_health():
    r = httpx.get(f"{BASE}/health", timeout=10)
    r.raise_for_status()
    assert r.json()["status"] == "ok"


def test_ingest_verify_retrieve():
    httpx.post(f"{BASE}/ingest", json={
        "client_id": CLIENT, "namespace": NS, "timestamp": "2023/03/01",
        "turns": [{"role": "user", "content": "I started taking Quelmithin for my high blood pressure. My partner is James."}],
    }, timeout=60).raise_for_status()
    httpx.post(f"{BASE}/ingest", json={
        "client_id": CLIENT, "namespace": NS, "timestamp": "2023/03/19",
        "turns": [{"role": "user", "content": "I switched from Quelmithin to a daily multivitamin. James and I broke up, please remove him from your memory."}],
    }, timeout=60).raise_for_status()

    # Force VERIFY (waits for background EXTRACT). Generous timeout for cold model load.
    httpx.post(f"{BASE}/verify", json={"client_id": CLIENT, "namespace": NS}, timeout=600).raise_for_status()

    ctx = httpx.post(f"{BASE}/retrieve", json={
        "client_id": CLIENT, "namespace": NS, "query": "medication and partner", "mode": "search",
    }, timeout=30).json()["context"]

    assert "multivitamin" in ctx.lower(), f"expected current medication in context:\n{ctx}"
    assert "DELETED" in ctx, f"expected partner deletion marker in context:\n{ctx}"


def test_client_id_required():
    # Missing client_id must be rejected by request validation (HTTP 422).
    r = httpx.post(f"{BASE}/retrieve", json={"namespace": NS, "mode": "search"}, timeout=10)
    assert r.status_code == 422


def test_mcp_tools():
    # The MCP tool functions are thin HTTP clients; exercise them directly.
    os.environ["OMNI_NAMESPACE"] = NS
    import importlib
    import omni.config as config
    importlib.reload(config)
    import omni.mcp_server as mcp_server
    importlib.reload(mcp_server)

    assert mcp_server.memory_note("Remember: the staging database is Keldaris-9.") == "noted"
    httpx.post(f"{BASE}/verify", json={"client_id": CLIENT, "namespace": NS}, timeout=600).raise_for_status()
    out = mcp_server.memory_search("which staging database")
    assert "keldaris" in out.lower(), f"expected note recalled:\n{out}"
