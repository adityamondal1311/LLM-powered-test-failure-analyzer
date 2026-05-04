from __future__ import annotations

from analyzer.models.pipeline import (
    FailureCategory,
    InferenceResult,
    RootCauseHypothesis,
    ValidationResult,
)


def _check_schema_integrity(h: RootCauseHypothesis) -> list[str]:
    issues: list[str] = []
    if not h.summary.strip():
        issues.append("summary is empty")
    if not h.explanation.strip():
        issues.append("explanation is empty")
    if h.category not in FailureCategory.__members__.values():
        issues.append(f"unknown category: {h.category!r}")
    if not (0.0 <= h.confidence <= 1.0):
        issues.append(f"confidence out of range: {h.confidence}")
    return issues


def validate_result(
    result: InferenceResult,
    threshold: float = 0.65,
) -> ValidationResult:
    issues = _check_schema_integrity(result.hypothesis)
    schema_valid = len(issues) == 0
    confidence_passed = result.hypothesis.confidence >= threshold

    return ValidationResult(
        inference=result,
        schema_valid=schema_valid,
        confidence_passed=confidence_passed,
        issues=issues,
    )
