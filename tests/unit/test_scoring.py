"""Unit tests for pipeline/scoring.py."""

from __future__ import annotations

import pytest

from analyzer.models.pipeline import (
    FailureCategory,
    FallbackSource,
    InferenceResult,
    RootCauseHypothesis,
    ValidationResult,
)
from analyzer.pipeline.scoring import _compute_rank_score, _is_actionable, score_result


# ---------------------------------------------------------------------------
# _is_actionable
# ---------------------------------------------------------------------------


def test_is_actionable_true_with_enough_words() -> None:
    assert _is_actionable("Check the assertion logic and fix the expected value") is True


def test_is_actionable_false_with_too_few_words() -> None:
    assert _is_actionable("fix it") is False


def test_is_actionable_false_with_empty_string() -> None:
    assert _is_actionable("") is False


def test_is_actionable_boundary_exactly_five_words() -> None:
    assert _is_actionable("one two three four five") is True


def test_is_actionable_boundary_four_words() -> None:
    assert _is_actionable("one two three four") is False


# ---------------------------------------------------------------------------
# _compute_rank_score
# ---------------------------------------------------------------------------


def _hypothesis(confidence: float, fix_hint: str) -> RootCauseHypothesis:
    return RootCauseHypothesis(
        category=FailureCategory.ASSERTION_ERROR,
        summary="test summary",
        explanation="test explanation",
        fix_hint=fix_hint,
        confidence=confidence,
        is_flaky=False,
    )


def test_rank_score_formula_all_bonuses() -> None:
    h = _hypothesis(0.9, "Check assertion logic and expected value in test")
    score = _compute_rank_score(h, schema_valid=True)
    expected = round(0.9 * 0.6 + 1.0 * 0.2 + 1.0 * 0.2, 4)
    assert score == pytest.approx(expected)


def test_rank_score_no_schema_validity() -> None:
    h = _hypothesis(0.8, "Fix the test data and schema assertions here")
    score = _compute_rank_score(h, schema_valid=False)
    expected = round(0.8 * 0.6 + 0.0 + 1.0 * 0.2, 4)
    assert score == pytest.approx(expected)


def test_rank_score_not_actionable() -> None:
    h = _hypothesis(0.8, "")
    score = _compute_rank_score(h, schema_valid=True)
    expected = round(0.8 * 0.6 + 0.2 + 0.0, 4)
    assert score == pytest.approx(expected)


def test_rank_score_maximum() -> None:
    h = _hypothesis(1.0, "Make sure to check the test fixture and expected values")
    score = _compute_rank_score(h, schema_valid=True)
    assert score == pytest.approx(1.0)


def test_rank_score_minimum() -> None:
    h = _hypothesis(0.0, "")
    score = _compute_rank_score(h, schema_valid=False)
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_result
# ---------------------------------------------------------------------------


def _make_validation(
    sample_parsed_failure: object,
    *,
    confidence: float = 0.90,
    fix_hint: str = "Run pip install and verify the dependency list",
    schema_valid: bool = True,
    confidence_passed: bool = True,
    fallback_used: bool = False,
    fallback_source: FallbackSource = FallbackSource.LLM,
) -> ValidationResult:
    h = RootCauseHypothesis(
        category=FailureCategory.ASSERTION_ERROR,
        summary="Test assertion failed",
        explanation="Values did not match.",
        fix_hint=fix_hint,
        confidence=confidence,
        is_flaky=False,
    )
    inference = InferenceResult(
        parsed_failure=sample_parsed_failure,  # type: ignore[arg-type]
        hypothesis=h,
        model_id="claude-sonnet-4-6",
        latency_ms=200.0,
        input_tokens=100,
        output_tokens=50,
        cache_hit_tokens=80,
        fallback_used=fallback_used,
        fallback_source=fallback_source,
    )
    return ValidationResult(
        inference=inference,
        schema_valid=schema_valid,
        confidence_passed=confidence_passed,
        issues=[],
    )


def test_score_result_routed_to_llm(sample_parsed_failure: object) -> None:
    v = _make_validation(sample_parsed_failure, fallback_used=False)
    scored = score_result(v)
    assert scored.routed_to == FallbackSource.LLM


def test_score_result_routed_to_heuristic_on_fallback(sample_parsed_failure: object) -> None:
    v = _make_validation(
        sample_parsed_failure,
        fallback_used=True,
        fallback_source=FallbackSource.HEURISTIC,
        confidence=0.40,
    )
    scored = score_result(v)
    assert scored.routed_to == FallbackSource.HEURISTIC


def test_score_result_routed_heuristic_when_confidence_failed(
    sample_parsed_failure: object,
) -> None:
    v = _make_validation(
        sample_parsed_failure,
        confidence=0.40,
        confidence_passed=False,
        fallback_used=False,
        fallback_source=FallbackSource.HEURISTIC,
    )
    scored = score_result(v)
    assert scored.routed_to == FallbackSource.HEURISTIC


def test_score_result_final_confidence(sample_parsed_failure: object) -> None:
    v = _make_validation(sample_parsed_failure, confidence=0.87)
    scored = score_result(v)
    assert scored.final_confidence == pytest.approx(0.87)


def test_score_result_actionable_with_good_hint(sample_parsed_failure: object) -> None:
    v = _make_validation(
        sample_parsed_failure,
        fix_hint="Run pip install and verify the dependency list",
    )
    scored = score_result(v)
    assert scored.actionable is True


def test_score_result_not_actionable_with_empty_hint(sample_parsed_failure: object) -> None:
    v = _make_validation(sample_parsed_failure, fix_hint="")
    scored = score_result(v)
    assert scored.actionable is False


def test_score_result_rank_score_is_float(sample_parsed_failure: object) -> None:
    v = _make_validation(sample_parsed_failure)
    scored = score_result(v)
    assert isinstance(scored.rank_score, float)
    assert 0.0 <= scored.rank_score <= 1.0
