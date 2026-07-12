"""Tests for the sentiment scorer (WS-B3).

All tests mock the transformers pipeline — no model downloads, no network.
"""
import sys

import pytest

from server.detectors.base import ScorerUnavailableError
from server.detectors.sentiment import CardiffSentimentScorer, overall_sentiment
from server.schemas import ParsedTurn


def _turns(*contents):
    return [ParsedTurn(role="user", content=c) for c in contents]


def _probs(neg, neu, pos):
    return [
        {"label": "negative", "score": neg},
        {"label": "neutral", "score": neu},
        {"label": "positive", "score": pos},
    ]


class FakePipe:
    def __init__(self, outputs):
        self.outputs = outputs

    def __call__(self, texts, batch_size=16):
        assert len(self.outputs) == len(texts)
        return self.outputs


def _make(outputs):
    scorer = CardiffSentimentScorer(model_name="fake-model", device="cpu")
    scorer._pipe = FakePipe(outputs)
    return scorer


def test_metadata():
    assert CardiffSentimentScorer(model_name="fake-model").name == "sentiment_encoder"


def test_negative_turn_scalar():
    scorer = _make([_probs(0.8, 0.1, 0.1)])
    out = scorer.score_turns(_turns("this is terrible"))
    assert out[0] == pytest.approx(-0.7, abs=1e-6)


def test_positive_turn_scalar():
    scorer = _make([_probs(0.05, 0.15, 0.8)])
    out = scorer.score_turns(_turns("this is wonderful"))
    assert out[0] == pytest.approx(0.75, abs=1e-6)


def test_batch_ordering_preserved():
    scorer = _make([_probs(0.8, 0.1, 0.1), _probs(0.05, 0.15, 0.8)])
    out = scorer.score_turns(_turns("bad", "good"))
    assert out[0] == pytest.approx(-0.7, abs=1e-6)
    assert out[1] == pytest.approx(0.75, abs=1e-6)


def test_construction_does_not_import_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    CardiffSentimentScorer(model_name="fake-model", device="cpu")


def test_unavailable_raises_scorer_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    scorer = CardiffSentimentScorer(model_name="fake-model", device="cpu")
    with pytest.raises(ScorerUnavailableError) as exc:
        scorer.score_turns(_turns("hello"))
    assert "requirements-ml.txt" in str(exc.value)


# --- overall_sentiment (pure function) ---

def test_overall_sentiment_empty():
    assert overall_sentiment([]) == 0.0


def test_overall_sentiment_three_entries_all_last_three():
    # weights [2,2,2] -> (2 + 2 - 2) / 6 = 1/3
    assert overall_sentiment([1.0, 1.0, -1.0]) == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_overall_sentiment_six_entries():
    # weights [1,1,1,2,2,2] -> (0+0+0+2+2-2)/9 = 2/9
    result = overall_sentiment([0.0, 0.0, 0.0, 1.0, 1.0, -1.0])
    assert result == pytest.approx(2.0 / 9.0, abs=1e-6)


def test_overall_sentiment_clamped():
    assert overall_sentiment([1.0]) == pytest.approx(1.0)
    assert overall_sentiment([-1.0, -1.0]) == pytest.approx(-1.0)
