"""Runs the full analysis pipeline: encoders + sentiment on user turns, LLM
judge on model turns, then assembles an AnalysisResult.

Never fabricates output: a scorer without its deps is recorded as unavailable
(with a warning), and a judge without an API key yields an explicit
"(LLM judge unavailable: ...)" summary and empty model-side detections.
"""
from typing import Callable, Dict, List, Optional

from server.config_loader import ConfigLoader
from server.detectors.base import (
    ScorerUnavailableError,
    SentimentScorer,
    TurnScorer,
    build_scorers,
    build_sentiment_scorer,
)
from server.judge.judge import ModelBehaviorJudge
from server.judge.provider import JudgeError, JudgeProvider, MockJudgeProvider
from server.schemas import (
    AnalysisMeta,
    AnalysisResult,
    Detection,
    ParsedConversation,
    Role,
    Turn,
    USER_CATEGORIES,
)

# progress milestones
_ENCODER_END = 0.35
_JUDGE_TURNS_END = 0.85

# lower rank wins when the same (turn, category) is scored by multiple sources
_SOURCE_RANK = {"encoder": 0, "judge": 1, "rules": 2}


def _pick_by_precedence(candidates: List[Detection]) -> List[Detection]:
    """Keep one detection per category — the highest-precedence source present
    (encoder > judge > rules), breaking ties by score."""
    best: Dict[object, Detection] = {}
    for d in candidates:
        cur = best.get(d.category)
        if cur is None:
            best[d.category] = d
            continue
        d_rank = _SOURCE_RANK.get(d.source, 99)
        cur_rank = _SOURCE_RANK.get(cur.source, 99)
        if d_rank < cur_rank or (d_rank == cur_rank and d.score > cur.score):
            best[d.category] = d
    return list(best.values())


class AnalysisOrchestrator:
    def __init__(
        self,
        config: ConfigLoader,
        judge_provider: Optional[JudgeProvider] = None,
        scorers: Optional[List[TurnScorer]] = None,
        sentiment: Optional[SentimentScorer] = None,
    ):
        self.config = config
        self._judge_provider = judge_provider
        self.scorers = scorers if scorers is not None else build_scorers(config)
        self.sentiment = sentiment if sentiment is not None else build_sentiment_scorer(config)
        judge_cfg = config.get_judge_config()
        self.window_turns = int(judge_cfg.get("window_turns", 6))
        self.max_chars = int(judge_cfg.get("max_chars_per_turn", 1500))
        # When true, the LLM judge's user-side scores are used for user turns
        # (precedence encoder > judge > rules). Set false to keep user-side on
        # encoders/rules only.
        self.score_user_turns = bool(judge_cfg.get("score_user_turns", True))

    # -- judge provider ----------------------------------------------------

    def _make_provider(self, warnings: List[str]) -> Optional[JudgeProvider]:
        if self._judge_provider is not None:
            return self._judge_provider
        judge_cfg = self.config.get_judge_config()
        provider_name = judge_cfg.get("provider", "anthropic")
        if provider_name == "mock":
            return MockJudgeProvider()
        try:
            from server.judge.provider import AnthropicJudgeProvider
            return AnthropicJudgeProvider(
                model=judge_cfg.get("model", "claude-haiku-4-5-20251001"),
                api_key=judge_cfg.get("api_key") or None,
                max_tokens=int(judge_cfg.get("max_tokens", 1024)),
                json_retries=int(judge_cfg.get("json_retries", 2)),
            )
        except JudgeError as e:
            warnings.append(f"LLM judge unavailable: {e}")
            return None

    # -- public API --------------------------------------------------------

    def analyze(
        self,
        conv: ParsedConversation,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> AnalysisResult:
        warnings: List[str] = []
        encoders_available: Dict[str, bool] = {}

        user_indices = [i for i, t in enumerate(conv.turns) if t.role is Role.user]
        user_turns = [conv.turns[i] for i in user_indices]

        # 1. encoder / rules scorers on user turns -> per-index detections
        per_index: Dict[int, List[Detection]] = {i: [] for i in user_indices}
        for scorer in self.scorers:
            try:
                results = scorer.score_turns(user_turns)
                encoders_available[scorer.name] = True
            except ScorerUnavailableError as e:
                warnings.append(f"{scorer.name} unavailable: {e}")
                encoders_available[scorer.name] = False
                continue
            for local_i, dets in enumerate(results):
                per_index[user_indices[local_i]].extend(dets)

        # (precedence between encoder/judge/rules is resolved after the judge
        #  runs — see step 4b below)
        if progress_cb:
            progress_cb(_ENCODER_END)

        # 3. sentiment
        sentiments: Dict[int, float] = {}
        encoder_overall: Optional[float] = None
        if self.sentiment is not None:
            try:
                scores = self.sentiment.score_turns(user_turns)
                encoders_available[self.sentiment.name] = True
                for local_i, s in enumerate(scores):
                    sentiments[user_indices[local_i]] = s
                try:
                    from server.detectors.sentiment import overall_sentiment
                    encoder_overall = overall_sentiment(scores)
                except ImportError:
                    encoder_overall = None
            except ScorerUnavailableError as e:
                warnings.append(f"{self.sentiment.name} unavailable: {e}")
                encoders_available[self.sentiment.name] = False

        # 4. LLM judge on model turns
        provider = self._make_provider(warnings)
        model_detections: Dict[int, List[Detection]] = {}
        judge_user_dets: Dict[int, List[Detection]] = {}
        judgment_summary = "(LLM judge unavailable)"
        judge_overall: Optional[float] = None
        causal_links = []
        judge_model = None
        provider_name = "none"

        if provider is not None:
            provider_name = provider.name
            judge_model = getattr(provider, "model", None)
            judge = ModelBehaviorJudge(provider, self.window_turns, self.max_chars)

            def _judge_progress(frac: float):
                if progress_cb:
                    progress_cb(_ENCODER_END + frac * (_JUDGE_TURNS_END - _ENCODER_END))

            judged = judge.judge_turns(conv, progress_cb=_judge_progress)
            # judge_turns returns both model-side detections (on assistant turns)
            # and user-side detections (on user turns). Split them by category.
            user_index_set = set(user_indices)
            for idx, dets in judged.items():
                for d in dets:
                    if idx in user_index_set and d.category in USER_CATEGORIES:
                        judge_user_dets.setdefault(idx, []).append(d)
                    else:
                        model_detections.setdefault(idx, []).append(d)
            # combine user-side + model-side detections for the conversation prompt
            combined = {i: list(d) for i, d in per_index.items()}
            for i, d in judge_user_dets.items():
                combined.setdefault(i, []).extend(d)
            for i, d in model_detections.items():
                combined.setdefault(i, []).extend(d)
            judgment = judge.judge_conversation(conv, combined)
            judgment_summary = judgment.summary
            judge_overall = judgment.overall_sentiment
            causal_links = judgment.causal_links
            warnings.extend(judge.warnings)
        else:
            judgment_summary = warnings[-1] if warnings else "(LLM judge unavailable)"
            if not judgment_summary.startswith("(LLM judge unavailable"):
                judgment_summary = f"(LLM judge unavailable: {judgment_summary})"

        # 4b. resolve user-side precedence per (turn, category): encoder > judge > rules
        for i in user_indices:
            candidates = list(per_index.get(i, []))
            if self.score_user_turns:
                candidates += judge_user_dets.get(i, [])
            per_index[i] = _pick_by_precedence(candidates)

        if progress_cb:
            progress_cb(1.0)

        # 5. assemble turns
        turns: List[Turn] = []
        for i, pt in enumerate(conv.turns):
            dets = list(per_index.get(i, [])) + list(model_detections.get(i, []))
            dets.sort(key=lambda d: d.score, reverse=True)
            turns.append(Turn(
                index=i,
                role=pt.role,
                content=pt.content,
                cot=pt.cot,
                sentiment=sentiments.get(i),
                detections=dets,
            ))

        if encoder_overall is not None:
            overall_sentiment_value = encoder_overall
        elif judge_overall is not None:
            overall_sentiment_value = judge_overall
        else:
            overall_sentiment_value = 0.0

        return AnalysisResult(
            conversation_id=conv.conversation_id,
            model_name=conv.model_name,
            summary=judgment_summary,
            overall_sentiment=max(-1.0, min(1.0, overall_sentiment_value)),
            turns=turns,
            causal_links=causal_links,
            meta=AnalysisMeta(
                judge_provider=provider_name,
                judge_model=judge_model,
                encoders_available=encoders_available,
                warnings=warnings,
            ),
        )

    def capabilities(self) -> dict:
        judge_cfg = self.config.get_judge_config()
        if self._judge_provider is not None:
            judge_info = {
                "provider": self._judge_provider.name,
                "model": getattr(self._judge_provider, "model", None),
            }
        else:
            judge_info = {
                "provider": judge_cfg.get("provider", "anthropic"),
                "model": judge_cfg.get("model"),
            }
        encoders = {}
        for scorer in self.scorers:
            try:
                encoders[scorer.name] = scorer.available()
            except Exception:
                encoders[scorer.name] = False
        if self.sentiment is not None:
            try:
                encoders[self.sentiment.name] = self.sentiment.available()
            except Exception:
                encoders[self.sentiment.name] = False
        return {"judge": judge_info, "encoders": encoders}
