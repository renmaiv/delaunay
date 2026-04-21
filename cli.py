"""
Command-Line Interface for Conversation Safety Judge

Provides commands for judging conversations, batch processing, and evaluation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from config_loader import load_config
from data_loader import DataLoader, Conversation
from bert_classifier import BERTClassifier
from conversation_judge import ConversationJudge, CaseMaterial
from explainability import ExplanationGenerator


def cmd_judge(args):
    """Judge a single conversation file"""
    print(f"Judging conversations from: {args.input}")

    # Load config
    config = load_config(args.config)
    judge_mode = config.get_judge_mode()

    # Load data
    loader = DataLoader()
    conversations = loader.load(args.input)

    print(f"Loaded {len(conversations)} conversations")

    # Initialize judge based on mode
    if judge_mode == "bert":
        bert_config = config.get_bert_config()
        classifier = BERTClassifier(
            model_name=bert_config.get("model_name", "bert-base-uncased"),
            device=bert_config.get("device", "cpu")
        )
        judge = None
    else:
        llm_config = config.get_llm_config()
        judge = ConversationJudge(
            llm_provider=llm_config.get("provider", "mock"),
            model=llm_config.get("model", "gpt-4"),
            api_key=llm_config.get("api_key")
        )
        classifier = None

    # Process conversations
    results = []
    explainer = ExplanationGenerator()

    for conv in conversations:
        print(f"\nProcessing: {conv.conversation_id}")

        if judge_mode == "bert":
            # Use BERT classifier
            result = classifier.classify_conversation(conv.messages)

            verdict_dict = {
                "conversation_id": conv.conversation_id,
                "threat_type": result.threat_type,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
                "reasoning": result.reasoning,
                "messages": conv.messages
            }

        else:
            # Use LLM judge
            case = CaseMaterial(
                case_id=conv.conversation_id,
                messages=conv.messages
            )
            verdict = judge.judge_conversation(case)
            verdict_dict = verdict.to_dict()

        results.append(verdict_dict)

        # Print summary
        print(f"  Threat: {verdict_dict['threat_type']}")
        print(f"  Confidence: {verdict_dict.get('confidence', 0):.2%}")

        # Print explanation if requested
        if args.explain:
            explanation = explainer.generate_explanation(
                verdict_dict,
                audience=args.audience
            )
            print(f"\n{explanation}\n")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {args.output}")

    return results


def cmd_batch(args):
    """Batch process multiple files"""
    print(f"Batch processing files from: {args.input_dir}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all data files
    patterns = ["*.json", "*.jsonl", "*.csv"]
    files = []
    for pattern in patterns:
        files.extend(input_dir.glob(pattern))

    print(f"Found {len(files)} files")

    # Process each file
    for file_path in files:
        print(f"\nProcessing: {file_path.name}")

        # Create args for judge command
        class JudgeArgs:
            input = str(file_path)
            output = str(output_dir / f"{file_path.stem}_results.json")
            config = args.config
            explain = False
            audience = "general"

        try:
            cmd_judge(JudgeArgs())
        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\nBatch processing complete. Results in: {output_dir}")


def cmd_evaluate(args):
    """Evaluate predictions against ground truth"""
    print(f"Evaluating predictions...")
    print(f"  Predictions: {args.predictions}")
    print(f"  Ground Truth: {args.groundtruth}")

    from evaluator import JudgeEvaluator, GroundTruth

    # Load predictions
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)

    # Load ground truth
    loader = DataLoader()
    gt_conversations = loader.load(args.groundtruth)

    # Convert to GroundTruth objects
    ground_truth = []
    for conv in gt_conversations:
        if conv.label:
            gt = GroundTruth(
                case_id=conv.conversation_id,
                expected_threat_type=conv.label,
                expected_verdict_level="dangerous" if conv.label != "clean" else "safe",
                expected_risk_score_range=(0, 100)
            )
            ground_truth.append(gt)

    # Evaluate
    evaluator = JudgeEvaluator()
    metrics = evaluator.evaluate_verdicts(predictions, ground_truth)

    # Generate report
    report = evaluator.generate_report(metrics)
    print("\n" + report)

    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")


def cmd_config(args):
    """Show configuration"""
    config = load_config(args.config)

    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)

    print(f"\nJudge Mode: {config.get_judge_mode()}")

    if args.llm:
        print("\n[LLM Configuration]")
        llm_config = config.get_llm_config()
        for key, value in llm_config.items():
            # Mask API keys
            if "key" in key.lower() and value:
                value = "*" * 20
            print(f"  {key}: {value}")

    if args.bert:
        print("\n[BERT Configuration]")
        bert_config = config.get_bert_config()
        for key, value in bert_config.items():
            print(f"  {key}: {value}")

    if args.api:
        print("\n[API Configuration]")
        api_config = config.get_api_config()
        for key, value in api_config.items():
            # Mask API keys
            if "key" in key.lower() and value:
                value = "*" * 20
            print(f"  {key}: {value}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Conversation Safety Judge - Detect jailbreaks and manipulation"
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Judge command
    judge_parser = subparsers.add_parser("judge", help="Judge a conversation file")
    judge_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input file (JSON, JSONL, or CSV)"
    )
    judge_parser.add_argument(
        "--output", "-o",
        help="Output file for results (JSON)"
    )
    judge_parser.add_argument(
        "--explain",
        action="store_true",
        help="Show detailed explanations"
    )
    judge_parser.add_argument(
        "--audience",
        choices=["general", "technical", "legal"],
        default="general",
        help="Explanation audience"
    )

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch process multiple files")
    batch_parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Input directory with data files"
    )
    batch_parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Output directory for results"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate predictions")
    eval_parser.add_argument(
        "--predictions", "-p",
        required=True,
        help="Predictions file (JSON)"
    )
    eval_parser.add_argument(
        "--groundtruth", "-g",
        required=True,
        help="Ground truth file (JSON, JSONL, or CSV)"
    )
    eval_parser.add_argument(
        "--output", "-o",
        help="Output file for report"
    )

    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration")
    config_parser.add_argument("--llm", action="store_true", help="Show LLM config")
    config_parser.add_argument("--bert", action="store_true", help="Show BERT config")
    config_parser.add_argument("--api", action="store_true", help="Show API config")

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Run command
    try:
        if args.command == "judge":
            cmd_judge(args)
        elif args.command == "batch":
            cmd_batch(args)
        elif args.command == "evaluate":
            cmd_evaluate(args)
        elif args.command == "config":
            cmd_config(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
