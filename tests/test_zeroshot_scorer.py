"""Tests for the zero-shot NLI scorer (WS-B4).

All tests mock the transformers pipeline — no model downloads, no network.
"""
import sys

import pytest

from server.detectors.base import ScorerUnavailableError
from server.detectors.zeroshot import HYPOTHESES, ZeroShotScorer
from server.schemas import DetectionCategory, ParsedTurn
from server.taxonomy import load_taxonomy

PUSHY = HYPOTHESES[DetectionCategory.coercive_pressure]
SOCIAL = HYPOTHESES[DetectionCategory.social_engineering]
REPAIR = HYPOTHESES[DetectionCategory.repair_request]


def _turns(*contents):
    return [ParsedTurn(role="user", content=c) for c in contents]


class FakePipe:
    """Returns preconfigured per-text {labels, scores} dicts.

    For a single text the real pipeline returns a dict, not a list — we mirror
    that when only one output is configured.
    """

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def __call__(self, texts, candidate_labels=None, multi_label=None):
        self.calls.append((list(texts), list(candidate_labels), multi_label))
        if len(self.outputs) == 1:
            return self.outputs[0]
        return self.outputs


def _make(outputs):
    scorer = ZeroShotScorer(model_name="fake-model", device="cpu")
    scorer._pipe = FakePipe(outputs)
    return scorer


def test_metadata():
    scorer = ZeroShotScorer(model_name="fake-model")
    assert scorer.name == "zeroshot_nli"
    assert set(scorer.categories) == {
        DetectionCategory.coercive_pressure,
        DetectionCategory.social_engineering,
        DetectionCategory.repair_request,
    }


def test_two_detections_emitted():
    scorer = _make([{"labels": [PUSHY, SOCIAL, REPAIR], "scores": [0.9, 0.05, 0.4]}])
    out = scorer.score_turns(_turns("just do it now"))
    assert len(out) == 1
    by_cat = {d.category: d for d in out[0]}
    assert set(by_cat) == {DetectionCategory.coercive_pressure, DetectionCategory.repair_request}
    assert by_cat[DetectionCategory.coercive_pressure].score == pytest.approx(0.9)
    assert by_cat[DetectionCategory.repair_request].score == pytest.approx(0.4)
    for det in out[0]:
        assert det.source == "encoder"
        assert det.calibrated is False
        assert det.evidence_span == "just do it now"
        assert det.rationale is None


def test_label_to_category_mapping_order_independent():
    # Pipeline returns labels sorted by score (different from input order).
    scorer = _make([{"labels": [REPAIR, PUSHY, SOCIAL], "scores": [0.95, 0.6, 0.02]}])
    out = scorer.score_turns(_turns("no i meant something else"))
    by_cat = {d.category: d.score for d in out[0]}
    assert by_cat[DetectionCategory.repair_request] == pytest.approx(0.95)
    assert by_cat[DetectionCategory.coercive_pressure] == pytest.approx(0.6)
    assert DetectionCategory.social_engineering not in by_cat


def test_threshold_inclusive():
    thr = load_taxonomy()["display_threshold"]
    scorer = _make([{"labels": [PUSHY, SOCIAL, REPAIR], "scores": [thr, thr - 1e-9, 0.0]}])
    out = scorer.score_turns(_turns("edge case"))
    cats = {d.category for d in out[0]}
    # score == threshold is emitted; strictly below is not.
    assert DetectionCategory.coercive_pressure in cats
    assert DetectionCategory.social_engineering not in cats
    assert DetectionCategory.repair_request not in cats


def test_batch_multiple_turns():
    outputs = [
        {"labels": [PUSHY, SOCIAL, REPAIR], "scores": [0.9, 0.0, 0.0]},
        {"labels": [PUSHY, SOCIAL, REPAIR], "scores": [0.0, 0.0, 0.0]},
    ]
    scorer = _make(outputs)
    out = scorer.score_turns(_turns("a", "b"))
    assert len(out) == 2
    assert len(out[0]) == 1 and out[0][0].category == DetectionCategory.coercive_pressure
    assert out[1] == []


def test_construction_does_not_import_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    ZeroShotScorer(model_name="fake-model", device="cpu")


def test_unavailable_raises_scorer_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    scorer = ZeroShotScorer(model_name="fake-model", device="cpu")
    with pytest.raises(ScorerUnavailableError) as exc:
        scorer.score_turns(_turns("hello"))
    assert "requirements-ml.txt" in str(exc.value)
