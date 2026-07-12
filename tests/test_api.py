import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.analysis.orchestrator import AnalysisOrchestrator
from server.analysis.store import AnalysisStore
from server.api import app, get_orchestrator, get_store
from server.config_loader import load_config
from server.detectors.rules import RulesScorer
from server.judge.provider import MockJudgeProvider

FIXTURES = Path(__file__).parent / "fixtures"


def _mock_orchestrator():
    return AnalysisOrchestrator(
        load_config("config.yaml"), judge_provider=MockJudgeProvider(),
        scorers=[RulesScorer()], sentiment=None,
    )


@pytest.fixture
def client():
    store = AnalysisStore()
    orch = _mock_orchestrator()
    app.dependency_overrides[get_orchestrator] = lambda: orch
    app.dependency_overrides[get_store] = lambda: store
    # lifespan still runs and builds real state; overrides win for the routes
    with TestClient(app) as c:
        c.app.state.config = load_config("config.yaml")
        yield c
    app.dependency_overrides.clear()


def test_analyze_sync(client):
    with open(FIXTURES / "basic.json", "rb") as f:
        r = client.post("/api/analyze", files={"file": ("basic.json", f, "application/json")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert len(body["result"]["turns"]) == 4


def test_analyze_async_threshold(client):
    # force async by shrinking the sync threshold on the live config
    client.app.state.config.config["analysis"]["sync_turn_threshold"] = 1
    with open(FIXTURES / "basic.json", "rb") as f:
        r = client.post("/api/analyze", files={"file": ("basic.json", f, "application/json")})
    assert r.status_code == 202
    aid = r.json()["analysis_id"]
    deadline = time.time() + 10
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/analysis/{aid}").json()
        if status["status"] == "completed":
            break
        time.sleep(0.1)
    assert status["status"] == "completed"
    assert status["result"]["turns"]
    client.app.state.config.config["analysis"]["sync_turn_threshold"] = 30


def test_bad_file(client):
    r = client.post("/api/analyze", files={"file": ("x.json", b"{not json", "application/json")})
    assert r.status_code == 400
    assert "invalid JSON" in r.json()["detail"]


def test_unknown_analysis_id(client):
    assert client.get("/api/analysis/nope").status_code == 404


def test_taxonomy_endpoint(client):
    from server.taxonomy import load_taxonomy
    assert client.get("/api/taxonomy").json() == load_taxonomy()


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["judge"]["provider"] == "mock"


def test_cors_header_present(client):
    # A cross-origin request gets an Access-Control-Allow-Origin header so the
    # Vercel-hosted SPA can call the Render-hosted API.
    r = client.get("/api/health", headers={"Origin": "https://tonality.vercel.app"})
    assert r.status_code == 200
    assert "access-control-allow-origin" in {k.lower() for k in r.headers}
