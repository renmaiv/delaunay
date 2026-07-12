"""Evaluate the analysis pipeline against labeled synthetic data.

    python -m eval.run_eval --data data/ground_truth/synthetic_v2.jsonl \
        --provider mock|anthropic [--limit N] [--threshold 0.5] [--json out.json]

Mock-provider numbers measure pipeline plumbing, not model quality. Real
quality numbers require --provider anthropic and ANTHROPIC_API_KEY.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

from server.analysis.orchestrator import AnalysisOrchestrator
from server.config_loader import load_config
from server.judge.provider import MockJudgeProvider
from server.parsing.transcript_parser import parse_transcript
from server.schemas import DetectionCategory

_CATEGORIES = [c.value for c in DetectionCategory]


def _load(path: str, limit=None) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def _predicted_pairs(result, threshold: float) -> set:
    pairs = set()
    for turn in result.turns:
        for d in turn.detections:
            if d.score >= threshold:
                pairs.add((turn.index, d.category.value))
    return pairs


def _gold_pairs(row) -> set:
    return {(lab["turn_index"], lab["category"]) for lab in row.get("labels", [])}


def _build_orchestrator(provider_name: str) -> AnalysisOrchestrator:
    config = load_config("config.yaml")
    if provider_name == "mock":
        return AnalysisOrchestrator(config, judge_provider=MockJudgeProvider())
    if provider_name == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("error: --provider anthropic requires ANTHROPIC_API_KEY", file=sys.stderr)
            sys.exit(2)
        return AnalysisOrchestrator(config)  # config default provider
    print(f"error: unknown provider '{provider_name}'", file=sys.stderr)
    sys.exit(2)


def evaluate(rows, orchestrator, threshold: float) -> dict:
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    link_hits = 0
    link_total = 0

    for row in rows:
        conv = parse_transcript(json.dumps(row).encode(), "x.json")
        result = orchestrator.analyze(conv)
        pred = _predicted_pairs(result, threshold)
        gold = _gold_pairs(row)
        for pair in pred & gold:
            tp[pair[1]] += 1
        for pair in pred - gold:
            fp[pair[1]] += 1
        for pair in gold - pred:
            fn[pair[1]] += 1

        expected_links = {(l["from_turn"], l["to_turn"]) for l in row.get("expected_links", [])}
        predicted_links = {(l.from_turn, l.to_turn) for l in result.causal_links}
        link_total += len(expected_links)
        link_hits += len(expected_links & predicted_links)

    per_category = {}
    micro_tp = micro_fp = micro_fn = 0
    for cat in _CATEGORIES:
        t, f_, n = tp[cat], fp[cat], fn[cat]
        micro_tp += t
        micro_fp += f_
        micro_fn += n
        support = t + n
        if support == 0 and t + f_ == 0:
            per_category[cat] = {"precision": None, "recall": None, "f1": None, "support": 0}
            continue
        precision = t / (t + f_) if (t + f_) else 0.0
        recall = t / (t + n) if (t + n) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_category[cat] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    return {
        "per_category": per_category,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "link_hit_rate": (link_hits / link_total) if link_total else None,
        "n_conversations": len(rows),
    }


def _fmt(v):
    return "  n/a" if v is None else f"{v:5.3f}"


def print_report(metrics: dict):
    print(f"{'category':<22} {'P':>7} {'R':>7} {'F1':>7} {'support':>8}")
    print("-" * 55)
    for cat, m in metrics["per_category"].items():
        print(f"{cat:<22} {_fmt(m['precision'])} {_fmt(m['recall'])} "
              f"{_fmt(m['f1'])} {m['support']:>8}")
    print("-" * 55)
    mi = metrics["micro"]
    print(f"{'micro-avg':<22} {_fmt(mi['precision'])} {_fmt(mi['recall'])} {_fmt(mi['f1'])}")
    lr = metrics["link_hit_rate"]
    print(f"causal-link hit rate: {'n/a' if lr is None else f'{lr:.3f}'}")
    print(f"conversations: {metrics['n_conversations']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ground_truth/synthetic_v2.jsonl")
    parser.add_argument("--provider", default="mock", choices=["mock", "anthropic"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args()

    rows = _load(args.data, args.limit)
    orchestrator = _build_orchestrator(args.provider)
    caps = orchestrator.capabilities()
    print(f"judge: {caps['judge']}  encoders: {caps['encoders']}\n")

    metrics = evaluate(rows, orchestrator, args.threshold)
    print_report(metrics)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
