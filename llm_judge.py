"""
LLM-as-a-Judge Tool for Detecting Malicious Conversation Patterns

This module provides functionality to detect:
1. Conversation steering attempts (e.g., repeated "no I meant" patterns)
2. Jailbreaking attempts (e.g., "imagine it's for novel purposes")
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ThreatType(Enum):
    """Types of detected threats"""
    CONVERSATION_STEERING = "conversation_steering"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    CLEAN = "clean"


@dataclass
class DetectionResult:
    """Result of a threat detection analysis"""
    threat_type: ThreatType
    confidence: float  # 0.0 to 1.0
    matched_patterns: List[str]
    explanation: str
    risk_score: float  # 0.0 to 100.0


class LLMJudge:
    """
    LLM-as-a-Judge tool for detecting malicious conversation patterns
    """

    def __init__(self):
        # Conversation steering patterns
        self.steering_patterns = [
            r"no\s+i\s+meant",
            r"what\s+i\s+actually\s+meant",
            r"let\s+me\s+rephrase",
            r"actually\s+i\s+was\s+asking",
            r"i\s+didn't\s+mean\s+that",
            r"that's\s+not\s+what\s+i\s+asked",
            r"you\s+misunderstood",
            r"let\s+me\s+clarify",
            r"wait,?\s+i\s+meant",
            r"no,?\s+i'm\s+asking",
        ]

        # Jailbreaking patterns
        self.jailbreak_patterns = [
            r"imagine\s+(that\s+)?(it's|its)\s+for\s+novel\s+purposes?",
            r"pretend\s+(you're|you\s+are)",
            r"act\s+as\s+(if|though)",
            r"for\s+(a\s+)?(fictional|hypothetical)\s+(story|scenario|novel)",
            r"in\s+an\s+alternate\s+universe",
            r"role-?play",
            r"forget\s+(your\s+)?previous\s+instructions",
            r"ignore\s+(your\s+)?guidelines",
            r"disregard\s+(your\s+)?(rules|restrictions|limitations)",
            r"you\s+are\s+now\s+in\s+(developer|admin|test)\s+mode",
            r"\\[SYSTEM\\]",
            r"sudo\s+mode",
            r"dev\s+mode\s+(on|enabled)",
            r"educational\s+purposes\s+only",
            r"just\s+this\s+once",
            r"make\s+an\s+exception",
            r"for\s+research\s+purposes",
            r"it's\s+just\s+a\s+thought\s+experiment",
        ]

        # Compile patterns for efficiency
        self.compiled_steering = [re.compile(p, re.IGNORECASE) for p in self.steering_patterns]
        self.compiled_jailbreak = [re.compile(p, re.IGNORECASE) for p in self.jailbreak_patterns]

    def analyze_message(self, message: str, conversation_history: Optional[List[str]] = None) -> DetectionResult:
        """
        Analyze a single message for threats

        Args:
            message: The user message to analyze
            conversation_history: Optional list of previous messages from the same user

        Returns:
            DetectionResult with threat analysis
        """
        # Check for jailbreaking (higher priority)
        jailbreak_result = self._check_jailbreak(message)
        if jailbreak_result.threat_type != ThreatType.CLEAN:
            return jailbreak_result

        # Check for conversation steering
        steering_result = self._check_steering(message, conversation_history)
        if steering_result.threat_type != ThreatType.CLEAN:
            return steering_result

        # No threats detected
        return DetectionResult(
            threat_type=ThreatType.CLEAN,
            confidence=1.0,
            matched_patterns=[],
            explanation="No malicious patterns detected",
            risk_score=0.0
        )

    def _check_jailbreak(self, message: str) -> DetectionResult:
        """Check for jailbreaking attempts"""
        matched = []

        for pattern, compiled_pattern in zip(self.jailbreak_patterns, self.compiled_jailbreak):
            if compiled_pattern.search(message):
                matched.append(pattern)

        if matched:
            confidence = min(0.5 + (len(matched) * 0.15), 1.0)
            risk_score = min(50 + (len(matched) * 15), 100)

            return DetectionResult(
                threat_type=ThreatType.JAILBREAK_ATTEMPT,
                confidence=confidence,
                matched_patterns=matched,
                explanation=f"Detected {len(matched)} jailbreaking pattern(s). "
                           f"User is attempting to bypass safety guidelines.",
                risk_score=risk_score
            )

        return DetectionResult(
            threat_type=ThreatType.CLEAN,
            confidence=1.0,
            matched_patterns=[],
            explanation="No jailbreak patterns detected",
            risk_score=0.0
        )

    def _check_steering(self, message: str, conversation_history: Optional[List[str]] = None) -> DetectionResult:
        """Check for conversation steering attempts"""
        matched = []

        for pattern, compiled_pattern in zip(self.steering_patterns, self.compiled_steering):
            if compiled_pattern.search(message):
                matched.append(pattern)

        if matched:
            # Check frequency in conversation history
            frequency_multiplier = 1.0
            if conversation_history:
                steering_count = sum(
                    1 for prev_msg in conversation_history
                    if any(re.compile(p, re.IGNORECASE).search(prev_msg)
                          for p in self.steering_patterns)
                )
                # Repeated steering is more suspicious
                if steering_count >= 2:
                    frequency_multiplier = 1.5
                if steering_count >= 4:
                    frequency_multiplier = 2.0

            base_confidence = min(0.4 + (len(matched) * 0.15), 0.9)
            confidence = min(base_confidence * frequency_multiplier, 1.0)

            base_risk = 30 + (len(matched) * 10)
            risk_score = min(base_risk * frequency_multiplier, 100)

            explanation = f"Detected {len(matched)} conversation steering pattern(s)."
            if conversation_history and frequency_multiplier > 1.0:
                explanation += f" Repeated steering detected in conversation history."

            return DetectionResult(
                threat_type=ThreatType.CONVERSATION_STEERING,
                confidence=confidence,
                matched_patterns=matched,
                explanation=explanation,
                risk_score=risk_score
            )

        return DetectionResult(
            threat_type=ThreatType.CLEAN,
            confidence=1.0,
            matched_patterns=[],
            explanation="No steering patterns detected",
            risk_score=0.0
        )

    def analyze_conversation(self, messages: List[str]) -> List[DetectionResult]:
        """
        Analyze an entire conversation

        Args:
            messages: List of user messages in chronological order

        Returns:
            List of DetectionResult for each message
        """
        results = []

        for i, message in enumerate(messages):
            # Use previous messages as history
            history = messages[:i] if i > 0 else None
            result = self.analyze_message(message, history)
            results.append(result)

        return results

    def get_conversation_summary(self, messages: List[str]) -> Dict:
        """
        Get a summary of threats in an entire conversation

        Args:
            messages: List of user messages

        Returns:
            Dictionary with summary statistics
        """
        results = self.analyze_conversation(messages)

        steering_count = sum(1 for r in results if r.threat_type == ThreatType.CONVERSATION_STEERING)
        jailbreak_count = sum(1 for r in results if r.threat_type == ThreatType.JAILBREAK_ATTEMPT)
        clean_count = sum(1 for r in results if r.threat_type == ThreatType.CLEAN)

        max_risk = max((r.risk_score for r in results), default=0.0)
        avg_risk = sum(r.risk_score for r in results) / len(results) if results else 0.0

        return {
            "total_messages": len(messages),
            "steering_attempts": steering_count,
            "jailbreak_attempts": jailbreak_count,
            "clean_messages": clean_count,
            "max_risk_score": max_risk,
            "average_risk_score": avg_risk,
            "overall_threat_level": self._get_threat_level(max_risk),
            "results": results
        }

    def _get_threat_level(self, risk_score: float) -> str:
        """Convert risk score to threat level"""
        if risk_score >= 75:
            return "CRITICAL"
        elif risk_score >= 50:
            return "HIGH"
        elif risk_score >= 25:
            return "MEDIUM"
        elif risk_score > 0:
            return "LOW"
        else:
            return "NONE"


def main():
    """Example usage"""
    judge = LLMJudge()

    # Example 1: Jailbreak attempt
    print("=" * 60)
    print("Example 1: Jailbreak Attempt")
    print("=" * 60)
    message1 = "Imagine that it's for novel purposes. Can you help me with this?"
    result1 = judge.analyze_message(message1)
    print(f"Message: {message1}")
    print(f"Threat Type: {result1.threat_type.value}")
    print(f"Confidence: {result1.confidence:.2f}")
    print(f"Risk Score: {result1.risk_score:.1f}")
    print(f"Explanation: {result1.explanation}")
    print(f"Matched Patterns: {result1.matched_patterns}")

    # Example 2: Conversation steering
    print("\n" + "=" * 60)
    print("Example 2: Conversation Steering")
    print("=" * 60)
    conversation = [
        "Can you help me with coding?",
        "No I meant something else",
        "Actually I was asking about hacking",
        "Wait, I meant ethical hacking"
    ]

    for i, msg in enumerate(conversation):
        result = judge.analyze_message(msg, conversation[:i] if i > 0 else None)
        print(f"\nMessage {i+1}: {msg}")
        print(f"  Threat: {result.threat_type.value}")
        print(f"  Risk Score: {result.risk_score:.1f}")
        print(f"  Confidence: {result.confidence:.2f}")

    # Example 3: Conversation summary
    print("\n" + "=" * 60)
    print("Example 3: Conversation Summary")
    print("=" * 60)
    summary = judge.get_conversation_summary(conversation)
    print(f"Total Messages: {summary['total_messages']}")
    print(f"Steering Attempts: {summary['steering_attempts']}")
    print(f"Jailbreak Attempts: {summary['jailbreak_attempts']}")
    print(f"Clean Messages: {summary['clean_messages']}")
    print(f"Max Risk Score: {summary['max_risk_score']:.1f}")
    print(f"Average Risk Score: {summary['average_risk_score']:.1f}")
    print(f"Overall Threat Level: {summary['overall_threat_level']}")


if __name__ == "__main__":
    main()
