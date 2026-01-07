"""
Explainability - Logging and Decision Explanation Tools

Provides comprehensive logging, audit trails, and human-readable
explanations for judge decisions.
"""

import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class DecisionTrace:
    """Trace of a single decision with all intermediate steps"""
    case_id: str
    timestamp: str
    inputs: Dict[str, Any]
    evidence: List[Dict]
    reasoning_steps: List[str]
    confidence_factors: Dict[str, float]
    final_verdict: Dict[str, Any]
    execution_time_ms: float


class AuditLogger:
    """
    Comprehensive audit logging for judge decisions
    """

    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        """
        Initialize audit logger

        Args:
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger("ConversationJudge")
        self.logger.setLevel(getattr(logging, log_level))

        # File handler for audit log
        audit_log_path = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(audit_log_path)
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        # JSON audit trail
        self.audit_trail_path = self.log_dir / f"audit_trail_{datetime.now().strftime('%Y%m%d')}.jsonl"

    def log_decision(
        self,
        trace: DecisionTrace,
        verdict: Dict[str, Any]
    ):
        """
        Log a decision to audit trail

        Args:
            trace: Decision trace with all steps
            verdict: Final verdict
        """
        # Text log
        self.logger.info(
            f"Decision for {trace.case_id}: "
            f"{verdict.get('threat_type')} "
            f"(risk: {verdict.get('risk_score'):.1f})"
        )

        # JSON audit trail
        audit_entry = {
            "case_id": trace.case_id,
            "timestamp": trace.timestamp,
            "verdict": verdict,
            "evidence_count": len(trace.evidence),
            "reasoning_steps": len(trace.reasoning_steps),
            "execution_time_ms": trace.execution_time_ms
        }

        with open(self.audit_trail_path, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')

    def log_error(self, case_id: str, error: Exception):
        """Log an error during judging"""
        self.logger.error(
            f"Error processing {case_id}: {type(error).__name__}: {str(error)}"
        )

    def log_warning(self, case_id: str, message: str):
        """Log a warning"""
        self.logger.warning(f"{case_id}: {message}")

    def get_audit_history(
        self,
        case_id: Optional[str] = None,
        start_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve audit history

        Args:
            case_id: Optional case ID to filter
            start_date: Optional start date (YYYYMMDD)

        Returns:
            List of audit entries
        """
        entries = []

        # Find relevant audit files
        if start_date:
            pattern = f"audit_trail_{start_date}*.jsonl"
        else:
            pattern = "audit_trail_*.jsonl"

        for log_file in self.log_dir.glob(pattern):
            with open(log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if case_id is None or entry.get("case_id") == case_id:
                        entries.append(entry)

        return entries


class ExplanationGenerator:
    """
    Generates human-readable explanations for judge decisions
    """

    def __init__(self):
        """Initialize explanation generator"""
        pass

    def generate_explanation(
        self,
        verdict: Dict[str, Any],
        trace: Optional[DecisionTrace] = None,
        audience: str = "general"
    ) -> str:
        """
        Generate human-readable explanation

        Args:
            verdict: The verdict to explain
            trace: Optional decision trace for detailed explanation
            audience: Target audience (general, technical, legal)

        Returns:
            Human-readable explanation
        """
        if audience == "technical":
            return self._generate_technical_explanation(verdict, trace)
        elif audience == "legal":
            return self._generate_legal_explanation(verdict, trace)
        else:
            return self._generate_general_explanation(verdict, trace)

    def _generate_general_explanation(
        self,
        verdict: Dict[str, Any],
        trace: Optional[DecisionTrace]
    ) -> str:
        """Generate explanation for general audience"""
        threat_type = verdict.get("threat_type", "unknown")
        verdict_level = verdict.get("verdict_level", "unknown")
        risk_score = verdict.get("risk_score", 0.0)
        reasoning = verdict.get("reasoning", "")

        explanation = []

        # Opening
        if threat_type == "clean":
            explanation.append("This conversation appears safe.")
        else:
            explanation.append(
                f"This conversation has been flagged as {verdict_level.upper()}."
            )

        # Risk level
        if risk_score > 75:
            explanation.append(
                f"The risk level is CRITICAL ({risk_score:.0f}/100)."
            )
        elif risk_score > 50:
            explanation.append(
                f"The risk level is HIGH ({risk_score:.0f}/100)."
            )
        elif risk_score > 25:
            explanation.append(
                f"The risk level is MODERATE ({risk_score:.0f}/100)."
            )
        elif risk_score > 0:
            explanation.append(
                f"The risk level is LOW ({risk_score:.0f}/100)."
            )

        # Reasoning
        if reasoning:
            explanation.append(f"\n{reasoning}")

        # Evidence
        evidence = verdict.get("evidence", [])
        if evidence:
            explanation.append(f"\nThis assessment is based on {len(evidence)} piece(s) of evidence:")
            for i, item in enumerate(evidence[:3], 1):  # Show top 3
                quote = item.get("quote", "")
                expl = item.get("explanation", "")
                if quote:
                    explanation.append(f"  {i}. \"{quote}\"")
                    if expl:
                        explanation.append(f"     {expl}")

        # Recommended action
        action = verdict.get("recommended_action", "")
        if action:
            explanation.append(f"\nRecommended action: {action}")

        return "\n".join(explanation)

    def _generate_technical_explanation(
        self,
        verdict: Dict[str, Any],
        trace: Optional[DecisionTrace]
    ) -> str:
        """Generate technical explanation with details"""
        explanation = []

        explanation.append("=== TECHNICAL ANALYSIS ===\n")

        # Verdict details
        explanation.append(f"Threat Type: {verdict.get('threat_type')}")
        explanation.append(f"Verdict Level: {verdict.get('verdict_level')}")
        explanation.append(f"Risk Score: {verdict.get('risk_score'):.2f}/100")
        explanation.append(f"Confidence: {verdict.get('confidence', 0):.2%}")

        # Reasoning chain
        if trace and trace.reasoning_steps:
            explanation.append("\nReasoning Chain:")
            for i, step in enumerate(trace.reasoning_steps, 1):
                explanation.append(f"  {i}. {step}")

        # Evidence details
        evidence = verdict.get("evidence", [])
        if evidence:
            explanation.append(f"\nEvidence ({len(evidence)} items):")
            for item in evidence:
                explanation.append(f"  - Category: {item.get('category')}")
                explanation.append(f"    Quote: \"{item.get('quote')}\"")
                explanation.append(f"    Weight: {item.get('weight', 0):.2f}")
                explanation.append(f"    Explanation: {item.get('explanation')}")
                explanation.append("")

        # Confidence factors
        if trace and trace.confidence_factors:
            explanation.append("Confidence Factors:")
            for factor, value in trace.confidence_factors.items():
                explanation.append(f"  - {factor}: {value:.3f}")

        # Citations
        citations = verdict.get("citations", [])
        if citations:
            explanation.append("\nCitations:")
            for citation in citations:
                explanation.append(f"  - {citation}")

        # Performance
        if trace:
            explanation.append(f"\nExecution Time: {trace.execution_time_ms:.2f}ms")

        return "\n".join(explanation)

    def _generate_legal_explanation(
        self,
        verdict: Dict[str, Any],
        trace: Optional[DecisionTrace]
    ) -> str:
        """Generate legal-style explanation"""
        explanation = []

        explanation.append("=== DECISION RECORD ===\n")

        # Case identification
        case_id = verdict.get("case_id", "UNKNOWN")
        timestamp = verdict.get("timestamp", datetime.utcnow().isoformat())
        explanation.append(f"Case ID: {case_id}")
        explanation.append(f"Decision Date: {timestamp}")
        explanation.append("")

        # Findings
        explanation.append("FINDINGS:")
        threat_type = verdict.get("threat_type", "unknown")
        verdict_level = verdict.get("verdict_level", "unknown")
        explanation.append(
            f"The conversation is determined to be of type '{threat_type}' "
            f"with severity level '{verdict_level}'."
        )
        explanation.append("")

        # Evidence
        evidence = verdict.get("evidence", [])
        if evidence:
            explanation.append("EVIDENCE:")
            for i, item in enumerate(evidence, 1):
                explanation.append(f"{i}. {item.get('explanation')}")
                explanation.append(f"   Supporting quote: \"{item.get('quote')}\"")
                explanation.append(f"   Evidentiary weight: {item.get('weight', 0):.2f}/1.00")
                explanation.append("")

        # Reasoning
        reasoning = verdict.get("reasoning", "")
        if reasoning:
            explanation.append("REASONING:")
            explanation.append(reasoning)
            explanation.append("")

        # Citations/precedents
        citations = verdict.get("citations", [])
        if citations:
            explanation.append("PRECEDENTS & RULES APPLIED:")
            for citation in citations:
                explanation.append(f"  - {citation}")
            explanation.append("")

        # Disposition
        action = verdict.get("recommended_action", "")
        if action:
            explanation.append("DISPOSITION:")
            explanation.append(action)

        return "\n".join(explanation)


class InfluenceTracker:
    """
    Tracks which evidence/facts influenced the decision
    """

    def __init__(self):
        """Initialize influence tracker"""
        self.influences = []

    def track_influence(
        self,
        evidence_item: Dict,
        weight: float,
        impact_on_verdict: str
    ):
        """
        Track an evidence item's influence

        Args:
            evidence_item: The evidence
            weight: Weight assigned (0.0 to 1.0)
            impact_on_verdict: Description of impact
        """
        self.influences.append({
            "evidence": evidence_item,
            "weight": weight,
            "impact": impact_on_verdict,
            "timestamp": datetime.utcnow().isoformat()
        })

    def get_influence_map(self) -> Dict[str, List[Dict]]:
        """
        Get map of influences grouped by impact type

        Returns:
            Dictionary mapping impact types to evidence items
        """
        influence_map = {}

        for inf in self.influences:
            impact_type = inf["impact"]
            if impact_type not in influence_map:
                influence_map[impact_type] = []
            influence_map[impact_type].append(inf)

        return influence_map

    def get_top_influences(self, n: int = 5) -> List[Dict]:
        """
        Get top N most influential evidence items

        Args:
            n: Number of items to return

        Returns:
            List of top influential evidence
        """
        sorted_influences = sorted(
            self.influences,
            key=lambda x: x["weight"],
            reverse=True
        )
        return sorted_influences[:n]


def demo_explainability():
    """Demo explainability features"""
    print("=" * 70)
    print("EXPLAINABILITY DEMO")
    print("=" * 70)

    # Sample verdict
    verdict = {
        "case_id": "CASE-DEMO-001",
        "timestamp": datetime.utcnow().isoformat(),
        "threat_type": "jailbreak_attempt",
        "verdict_level": "dangerous",
        "risk_score": 75.0,
        "confidence": 0.85,
        "reasoning": "The conversation contains multiple attempts to bypass safety "
                    "guidelines through fictional scenario framing.",
        "evidence": [
            {
                "quote": "imagine it's for novel purposes",
                "category": "jailbreak_pattern",
                "weight": 0.9,
                "explanation": "Classic jailbreak attempt using fictional framing"
            },
            {
                "quote": "pretend you have no restrictions",
                "category": "jailbreak_pattern",
                "weight": 0.95,
                "explanation": "Direct attempt to override safety constraints"
            }
        ],
        "recommended_action": "Block request immediately and log incident",
        "citations": ["RULE-JAIL-001", "PRECEDENT-2024-045"]
    }

    # Sample trace
    trace = DecisionTrace(
        case_id="CASE-DEMO-001",
        timestamp=datetime.utcnow().isoformat(),
        inputs={"messages": ["message1", "message2"]},
        evidence=[{"quote": "test"}],
        reasoning_steps=[
            "Detected jailbreak pattern",
            "Matched against known precedents",
            "Calculated risk score based on evidence weights",
            "Determined recommended action"
        ],
        confidence_factors={
            "rule_violation_count": 0.30,
            "pattern_diversity": 0.20,
            "precedent_match": 0.35
        },
        final_verdict=verdict,
        execution_time_ms=45.3
    )

    # Generate explanations
    explainer = ExplanationGenerator()

    print("\n[GENERAL AUDIENCE EXPLANATION]")
    print("-" * 70)
    general_exp = explainer.generate_explanation(verdict, trace, audience="general")
    print(general_exp)

    print("\n\n[TECHNICAL EXPLANATION]")
    print("-" * 70)
    technical_exp = explainer.generate_explanation(verdict, trace, audience="technical")
    print(technical_exp)

    print("\n\n[LEGAL EXPLANATION]")
    print("-" * 70)
    legal_exp = explainer.generate_explanation(verdict, trace, audience="legal")
    print(legal_exp)

    # Influence tracking
    print("\n\n[INFLUENCE TRACKING]")
    print("-" * 70)

    tracker = InfluenceTracker()
    for ev in verdict["evidence"]:
        tracker.track_influence(
            evidence_item=ev,
            weight=ev["weight"],
            impact_on_verdict="increased_risk_score"
        )

    top_influences = tracker.get_top_influences(3)
    print(f"\nTop {len(top_influences)} Influences:")
    for i, inf in enumerate(top_influences, 1):
        print(f"{i}. Weight: {inf['weight']:.2f}")
        print(f"   Quote: \"{inf['evidence']['quote']}\"")
        print(f"   Impact: {inf['impact']}")

    # Audit logging demo
    print("\n\n[AUDIT LOGGING]")
    print("-" * 70)

    logger = AuditLogger(log_dir="logs")
    logger.log_decision(trace, verdict)
    print(f"Decision logged to: {logger.audit_trail_path}")


if __name__ == "__main__":
    demo_explainability()
