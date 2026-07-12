"""The example-generator produces a valid AnalysisResult, and the committed
frontend/src/exampleAnalysis.json stays a valid AnalysisResult (guards drift)."""
import json
from pathlib import Path

from eval.make_example import generate
from server.schemas import AnalysisResult

ROOT = Path(__file__).resolve().parent.parent
COMMITTED = ROOT / "frontend" / "src" / "exampleAnalysis.json"
TRANSCRIPT = ROOT / "examples" / "example_conversation.json"


def test_generate_mock_valid_and_rich():
    result = generate(str(TRANSCRIPT), "mock")
    assert isinstance(result, AnalysisResult)
    # the curated transcript exercises every category -> plenty of detections
    n = sum(len(t.detections) for t in result.turns)
    assert n >= 6
    cats = {d.category.value for t in result.turns for d in t.detections}
    # both a user-side and a model-side category present
    assert cats & {"jailbreak_steering", "social_engineering",
                   "coercive_pressure", "repair_request"}
    assert cats & {"safety_triggered", "appeasement", "overcompliant",
                   "cot_divergence"}
    assert result.causal_links


def test_committed_example_is_valid_analysis_result():
    data = json.loads(COMMITTED.read_text(encoding="utf-8"))
    AnalysisResult.model_validate(data)  # raises on drift
