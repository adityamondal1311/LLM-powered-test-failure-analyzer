"""Unit tests for pipeline/validation.py."""

from __future__ import annotations

import pytest

from analyzer.models.pipeline import (
    FailureCategory,
    FallbackSource,
    InferenceResult,
    RootCauseHypothesis,
)
from analyzer.pipeline.validation import validate_result


def _make_inference(
    hypothesis: RootCauseHypothesis,
    sample_parsed_failure: object,
) -> InferenceResult:
    return InferenceResult(
        parsed_failure=sample_parsed_failure,  # type: ignore[arg-type]
        hypothesis=hypothesis,
        model_id="claude-sonnet-4-6",
        latency_ms=100.0,
        input_tokens=50,
        output_tokens=20,
        cache_hit_tokens=40,
        fallback_used=False,
        fallback_source=FallbackSource.LLM,
    )


def _good_hypothesis(**overrides: object) -> RootCauseHypothesis:
    base: dict = {
        "category": FailureCategory.ASSERTION_ERROR,
        "summary": "Test assertion failed",
        "explanation": "Values did not match as expected.",
        "fix_hint": "Check the assertion logic and expected values in the test.",
        "confidence": 0.90,
        "is_flaky": False,
    }
    base.update(overrides)
    return RootCauseHypothesis(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Schema valid / confidence passed
# ---------------------------------------------------------------------------


def test_valid_result_passes(sample_parsed_failure: object) -> None:
    h = _good_hypothesis()
    result = validate_result(_make_inference(h, sample_parsed_failure))
    assert result.schema_valid is True
    assert result.confidence_passed is True
    assert result.issues == []


def test_confidence_above_threshold_passes(sample_parsed_failure: object) -> None:
    h = _good_hypothesis(confidence=0.65)
    result = validate_result(_make_inference(h, sample_parsed_failure), threshold=0.65)
    assert result.confidence_passed is True


def test_confidence_below_threshold_fails(sample_parsed_failure: object) -> None:
    h = _good_hypothesis(confidence=0.40)
    result = validate_result(_make_inference(h, sample_parsed_failure), threshold=0.65)
    assert result.confidence_passed is False


# ---------------------------------------------------------------------------
# Schema integrity checks
# ---------------------------------------------------------------------------


def test_empty_summary_fails(sample_parsed_failure: object) -> None:
    h = _good_hypothesis(summary="   ")
    result = validate_result(_make_inference(h, sample_parsed_failure))
    assert result.schema_valid is False
    assert any("summary" in issue for issue in result.issues)


def test_empty_explanation_fails(sample_parsed_failure: object) -> None:
    h = _good_hypothesis(explanation="  ")
    result = validate_result(_make_inference(h, sample_parsed_failure))
    assert result.schema_valid is False
    assert any("explanation" in issue for issue in result.issues)


def test_multiple_issues_recorded(sample_parsed_failure: object) -> None:
    h = _good_hypothesis(summary="  ", explanation="  ")
    result = validate_result(_make_inference(h, sample_parsed_failure))
    assert result.schema_valid is False
    assert len(result.issues) == 2


def test_valid_category_passes(sample_parsed_failure: object) -> None:
    for cat in FailureCategory:
        h = _good_hypothesis(category=cat)
        result = validate_result(_make_inference(h, sample_parsed_failure))
        assert not any("unknown category" in issue for issue in result.issues)


# ---------------------------------------------------------------------------
# inference is passed through unchanged
# ---------------------------------------------------------------------------


def test_inference_preserved(sample_parsed_failure: object) -> None:
    h = _good_hypothesis()
    inference = _make_inference(h, sample_parsed_failure)
    result = validate_result(inference)
    assert result.inference is inference


# ---------------------------------------------------------------------------
# Custom threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threshold", [0.5, 0.65, 0.8])
def test_custom_threshold(sample_parsed_failure: object, threshold: float) -> None:
    h = _good_hypothesis(confidence=0.75)
    result = validate_result(_make_inference(h, sample_parsed_failure), threshold=threshold)
    assert result.confidence_passed == (0.75 >= threshold)
