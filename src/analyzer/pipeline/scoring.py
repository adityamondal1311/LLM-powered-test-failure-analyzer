from __future__ import annotations

from analyzer.models.pipeline import (
    FallbackSource,
    RootCauseHypothesis,
    ScoredResult,
    ValidationResult,
)

_MIN_ACTIONABLE_WORDS = 5


def _is_actionable(fix_hint: str) -> bool:
    return len(fix_hint.split()) >= _MIN_ACTIONABLE_WORDS


def _compute_rank_score(
    h: RootCauseHypothesis,
    schema_valid: bool,
) -> float:
    validity_bonus = 1.0 if schema_valid else 0.0
    fix_quality = 1.0 if _is_actionable(h.fix_hint) else 0.0
    return round(h.confidence * 0.6 + validity_bonus * 0.2 + fix_quality * 0.2, 4)


def score_result(validation: ValidationResult) -> ScoredResult:
    h = validation.inference.hypothesis
    rank_score = _compute_rank_score(h, validation.schema_valid)

    # If confidence gate failed, the actual routing is to heuristic — mark it
    routed_to = (
        validation.inference.fallback_source
        if validation.inference.fallback_used or not validation.confidence_passed
        else FallbackSource.LLM
    )

    return ScoredResult(
        validation=validation,
        final_confidence=h.confidence,
        rank_score=rank_score,
        actionable=_is_actionable(h.fix_hint),
        routed_to=routed_to,
    )
