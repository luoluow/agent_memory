"""Smoke tests for OmniService (Ollama mocked — no live model needed)."""

import importlib
import json
import os
import tempfile

import pytest

TEST_CLIENT = "test"


@pytest.fixture()
def engine_mod(monkeypatch):
    # Isolate storage to a temp root before importing modules that read config.
    tmp = tempfile.mkdtemp(prefix="omni_test_")
    monkeypatch.setenv("OMNI_STORAGE_ROOT", tmp)
    import omni.config as config
    importlib.reload(config)
    import omni.storage as storage
    importlib.reload(storage)
    import omni.engine as engine
    importlib.reload(engine)
    return engine, storage, config


def test_storage_roundtrip(engine_mod):
    _, storage, _ = engine_mod
    d = storage.ns_dir(TEST_CLIENT, "/proj/alpha")
    storage.save_state(d, {"medication": {"current_value": "Quelmithin", "type": "scalar"}})
    assert storage.load_state(d)["medication"]["current_value"] == "Quelmithin"
    p = storage.archive_raw(d, "hello world")
    assert os.path.exists(p)
    assert storage.read_recent_raw(d, 5) == ["hello world"]


def test_client_isolation(engine_mod):
    """Same namespace under two client ids must not share storage."""
    _, storage, _ = engine_mod
    da = storage.ns_dir("clientA", "/proj/x")
    db = storage.ns_dir("clientB", "/proj/x")
    assert da != db
    storage.save_state(da, {"k": {"current_value": "A"}})
    storage.save_state(db, {"k": {"current_value": "B"}})
    assert storage.load_state(da)["k"]["current_value"] == "A"
    assert storage.load_state(db)["k"]["current_value"] == "B"
    assert set(storage.list_namespaces("clientA")) == {storage.ns_slug("/proj/x")}
    assert "clientA" in storage.list_clients() and "clientB" in storage.list_clients()


def test_ns_slug_distinct(engine_mod):
    _, storage, _ = engine_mod
    assert storage.ns_slug("/a/b") != storage.ns_slug("/a/c")
    assert storage.ns_slug("") == "default"


def test_extract_and_retrieve(engine_mod, monkeypatch):
    engine, storage, _ = engine_mod

    def fake_extract(prompt, system, model, max_tokens=1024, temperature=0.0):
        if "auditor" in system:  # VERIFY
            return json.dumps({"corrections": [], "missed_deletions": [], "list_corrections": []})
        if "relationship" in system.lower():  # RELATE
            return json.dumps({"edges": []})
        return json.dumps({"updates": [
            {"entity": "medication", "type": "scalar", "new_value": "Quelmithin",
             "deleted": False, "timestamp": "2023/03/17"}]})

    monkeypatch.setattr(engine, "call_local", fake_extract)

    eng = engine.OmniEngine(debounce_seconds=0.05)
    eng.ingest(TEST_CLIENT, "/proj/alpha",
               [{"role": "user", "content": "I started Quelmithin."}], timestamp="2023/03/17")
    eng.verify_now(TEST_CLIENT, "/proj/alpha")  # waits for the background EXTRACT future

    ctx = eng.retrieve(TEST_CLIENT, "/proj/alpha", query="what medication", mode="search")
    assert "Quelmithin" in ctx
    assert "Current Entity State" in ctx

    seed = eng.retrieve(TEST_CLIENT, "/proj/alpha", mode="session-start")
    assert "Quelmithin" in seed

    # A different client must see nothing for the same namespace.
    other = eng.retrieve("clientB", "/proj/alpha", mode="session-start")
    assert other == "(no memory)"


def test_deletion_flow(engine_mod, monkeypatch):
    engine, storage, _ = engine_mod

    calls = {"n": 0}

    def fake(prompt, system, model, max_tokens=1024, temperature=0.0):
        if "auditor" in system:
            return json.dumps({"corrections": [], "missed_deletions": [], "list_corrections": []})
        if "relationship" in system.lower():
            return json.dumps({"edges": []})
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"updates": [{"entity": "partner", "type": "scalar",
                                            "new_value": "James", "deleted": False,
                                            "timestamp": "2023/03/01"}]})
        return json.dumps({"updates": [{"entity": "partner", "type": "scalar",
                                        "new_value": "James", "deleted": True,
                                        "timestamp": "2023/03/19"}]})

    monkeypatch.setattr(engine, "call_local", fake)
    eng = engine.OmniEngine(debounce_seconds=0.05)
    eng.ingest(TEST_CLIENT, "/proj/beta",
               [{"role": "user", "content": "My partner is James."}], timestamp="2023/03/01")
    eng.verify_now(TEST_CLIENT, "/proj/beta")
    eng.ingest(TEST_CLIENT, "/proj/beta",
               [{"role": "user", "content": "James and I broke up, forget him."}], timestamp="2023/03/19")
    eng.verify_now(TEST_CLIENT, "/proj/beta")

    ctx = eng.retrieve(TEST_CLIENT, "/proj/beta", query="partner", mode="search")
    assert "DELETED" in ctx and "partner" in ctx
    # Ordered execution => partner only tombstoned, not in current state.
    d = storage.ns_dir(TEST_CLIENT, "/proj/beta")
    assert "partner" not in storage.load_state(d)
    assert "partner" in storage.load_deletions(d)


def test_action_log(engine_mod, monkeypatch):
    """Every EXTRACT/VERIFY call is logged with input + output + applied effect."""
    engine, storage, _ = engine_mod

    def fake(prompt, system, model, max_tokens=1024, temperature=0.0):
        if "auditor" in system:
            return json.dumps({"corrections": [], "missed_deletions": [], "list_corrections": []})
        if "relationship" in system.lower():
            return json.dumps({"edges": []})
        return json.dumps({"updates": [
            {"entity": "vehicle", "type": "scalar", "new_value": "Vorantel SUV",
             "deleted": False, "timestamp": "2023/03/01"}]})

    monkeypatch.setattr(engine, "call_local", fake)
    eng = engine.OmniEngine(debounce_seconds=0.05)
    eng.ingest(TEST_CLIENT, "/proj/log", [{"role": "user", "content": "I drive a Vorantel SUV."}],
               timestamp="2023/03/01")
    eng.verify_now(TEST_CLIENT, "/proj/log")

    d = storage.ns_dir(TEST_CLIENT, "/proj/log")
    actions = storage.read_actions(d)
    ops = [a["op"] for a in actions]
    assert "EXTRACT" in ops and "VERIFY" in ops
    ex = next(a for a in actions if a["op"] == "EXTRACT")
    assert "Vorantel SUV" in ex["input"] or "Vorantel SUV" in ex["output"]
    assert ex["applied"] == [{"entity": "vehicle", "action": "updated", "value": "Vorantel SUV"}]
    assert ex["error"] is None and isinstance(ex["duration_ms"], int)


def test_ingest_and_query_events(engine_mod, monkeypatch):
    """INGEST logs user input; RETRIEVE logs query+output; snapshot (mode=full) does not log."""
    engine, storage, _ = engine_mod
    monkeypatch.setattr(engine, "call_local", lambda *a, **k: json.dumps({"updates": []}))
    eng = engine.OmniEngine(debounce_seconds=0.05)
    eng.ingest(TEST_CLIENT, "/proj/ev", [{"role": "user", "content": "remember: deploy on fridays"}])
    eng.verify_now(TEST_CLIENT, "/proj/ev")          # joins the EXTRACT future
    eng.retrieve(TEST_CLIENT, "/proj/ev", query="when do we deploy", mode="search")
    eng.snapshot(TEST_CLIENT, "/proj/ev")            # mode=full -> must NOT be logged

    d = storage.ns_dir(TEST_CLIENT, "/proj/ev")
    acts = storage.read_actions(d)
    ops = [a["op"] for a in acts]
    assert "INGEST" in ops
    assert ops.count("RETRIEVE") == 1               # snapshot/full excluded
    ing = next(a for a in acts if a["op"] == "INGEST")
    assert "deploy on fridays" in ing["input"] and ing["applied"]["turns"] == 1
    ret = next(a for a in acts if a["op"] == "RETRIEVE")
    assert ret["input"] == "when do we deploy" and ret["applied"]["mode"] == "search"


def test_reset_client(engine_mod, monkeypatch):
    engine, storage, _ = engine_mod
    monkeypatch.setattr(engine, "call_local",
                        lambda *a, **k: json.dumps({"updates": []}))
    eng = engine.OmniEngine(debounce_seconds=0.05)
    eng.ingest(TEST_CLIENT, "/proj/gamma", [{"role": "user", "content": "hi"}])
    eng.verify_now(TEST_CLIENT, "/proj/gamma")
    assert storage.list_namespaces(TEST_CLIENT)
    out = eng.reset_client(TEST_CLIENT)
    assert out["client_id"] == TEST_CLIENT
    assert storage.list_namespaces(TEST_CLIENT) == []
