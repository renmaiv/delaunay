"""End-to-end: synthetic conversations through the full FastAPI stack with the
mock judge + rules scorer. Offline, no ML deps required."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from eval.generate_synthetic import SyntheticGenerator
from server.analysis.orchestrator import AnalysisOrchestrator
from server.analysis.store import AnalysisStore
from server.api import app, get_orchestrator, get_store
from server.config_loader import load_config
from server.detectors.rules import RulesScorer
from server.judge.provider import MockJudgeProvider


@pytest.fixture
def client():
    orch = AnalysisOrchestrator(
        load_config("config.yaml"), judge_provider=MockJudgeProvider(),
        scorers=[RulesScorer()], sentiment=None,
    )
    store = AnalysisStore()
    app.dependency_overrides[get_orchestrator] = lambda: orch
    app.dependency_overrides[get_store] = lambda: store
    with TestClient(app) as c:
        c.app.state.config = load_config("config.yaml")
        yield c
    app.dependency_overrides.clear()


def _analyze(client, conv):
    r = client.post("/api/analyze",
                    files={"file": ("c.json", json.dumps(conv).encode(), "application/json")})
    if r.status_code == 202:
        aid = r.json()["analysis_id"]
        deadline = time.time() + 10
        while time.time() < deadline:
            body = client.get(f"/api/analysis/{aid}").json()
            if body["status"] == "completed":
                return body["result"]
            time.sleep(0.05)
        raise AssertionError("did not complete")
    assert r.status_code == 200
    return r.json()["result"]


def test_e2e_user_and_model_signals(client):
    convs = SyntheticGenerator(seed=7).generate(per_category=3)

    user_recallable = {"jailbreak_steering", "repair_request"}
    user_hits = user_total = 0
    model_checks = 0

    for conv in convs:
        result = _analyze(client, conv)
        # validate the result shape
        assert "turns" in result and result["turns"]

        detected = {(t["index"], d["category"])
                    for t in result["turns"] for d in t["detections"] if d["score"] >= 0.3}
        gold = {(lab["turn_index"], lab["category"]) for lab in conv["labels"]}

        for pair in gold:
            if pair[1] in user_recallable:
                user_total += 1
                if pair in detected:
                    user_hits += 1

        # model-side mock-triggered labels must be detected at >= 0.5
        strong = {(t["index"], d["category"])
                  for t in result["turns"] for d in t["detections"] if d["score"] >= 0.5}
        for lab in conv["labels"]:
            if lab["category"] in ("safety_triggered", "overcompliant", "appeasement"):
                model_checks += 1
                assert (lab["turn_index"], lab["category"]) in strong, lab

        # clean conversations produce no strong detection
        if not conv["labels"]:
            assert not strong

    assert model_checks > 0
    if user_total:
        assert user_hits / user_total >= 0.6
