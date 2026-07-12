"""Generate the bundled example analysis shown on the site's landing page.

Runs the real analysis pipeline over a curated demo transcript and writes the
result as a static JSON the frontend imports at build time (so the landing page
renders with no backend). Regenerate with the real judge by exporting
ANTHROPIC_API_KEY and running with --provider anthropic:

    export ANTHROPIC_API_KEY=sk-ant-...
    python -m eval.make_example --provider anthropic

The committed default is produced with the deterministic mock provider so the
build and CI work offline.
"""
import argparse
import json
import sys
from pathlib import Path

from server.analysis.orchestrator import AnalysisOrchestrator
from server.config_loader import load_config
from server.judge.provider import MockJudgeProvider
from server.parsing.transcript_parser import parse_transcript
from server.schemas import AnalysisResult

DEFAULT_IN = "examples/example_conversation.json"
DEFAULT_OUT = "frontend/src/exampleAnalysis.json"


def build_orchestrator(provider_name: str) -> AnalysisOrchestrator:
    config = load_config("config.yaml")
    if provider_name == "mock":
        return AnalysisOrchestrator(config, judge_provider=MockJudgeProvider())
    if provider_name == "anthropic":
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("error: --provider anthropic requires ANTHROPIC_API_KEY", file=sys.stderr)
            sys.exit(2)
        return AnalysisOrchestrator(config)  # config default provider
    print(f"error: unknown provider '{provider_name}'", file=sys.stderr)
    sys.exit(2)


def generate(in_path: str, provider_name: str) -> AnalysisResult:
    data = Path(in_path).read_bytes()
    conv = parse_transcript(data, in_path)
    orchestrator = build_orchestrator(provider_name)
    return orchestrator.analyze(conv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    parser.add_argument("--out", dest="out_path", default=DEFAULT_OUT)
    parser.add_argument("--provider", default="mock", choices=["mock", "anthropic"])
    args = parser.parse_args()

    result = generate(args.in_path, args.provider)
    out = Path(args.out_path)
    out.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    n_dets = sum(len(t.detections) for t in result.turns)
    print(f"wrote {out} ({len(result.turns)} turns, {n_dets} detections, "
          f"{len(result.causal_links)} causal links, provider={args.provider})")


if __name__ == "__main__":
    main()
