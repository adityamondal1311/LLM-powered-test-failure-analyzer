"""Unit tests for fallback/heuristics.py."""

from __future__ import annotations

import pytest

from analyzer.fallback.heuristics import classify_by_heuristic
from analyzer.models.pipeline import FailureCategory


def _classify(combined: str) -> object:
    """Helper: pass the same text as all three args (simple integration)."""
    return classify_by_heuristic(combined, combined, combined)


# ---------------------------------------------------------------------------
# Category detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_category",
    [
        ("AssertionError: assert 1 == 2", FailureCategory.ASSERTION_ERROR),
        ("assert x == y failed badly", FailureCategory.ASSERTION_ERROR),
        ("ModuleNotFoundError: No module named 'foo'", FailureCategory.IMPORT_ERROR),
        ("ImportError: cannot import name 'bar'", FailureCategory.IMPORT_ERROR),
        ("fixture 'db_session' not found error", FailureCategory.FIXTURE_ERROR),
        ("fixture setup failed unexpectedly", FailureCategory.FIXTURE_ERROR),
        ("TimeoutError: timed out", FailureCategory.TIMEOUT),
        ("operation timed out after 5s", FailureCategory.TIMEOUT),
        ("ConnectionError: failed to connect", FailureCategory.NETWORK_ERROR),
        ("SSLError: certificate verify failed", FailureCategory.NETWORK_ERROR),
        ("HTTPError: 503 service unavailable", FailureCategory.NETWORK_ERROR),
        ("ConnectionRefused on port 5432", FailureCategory.NETWORK_ERROR),
        ("KeyError: env var DATABASE_URL missing", FailureCategory.ENVIRONMENT_ERROR),
        ("os.environ access failed", FailureCategory.ENVIRONMENT_ERROR),
        ("FileNotFoundError: No such file or directory: 'data.csv'", FailureCategory.DATA_ERROR),
        ("schema mismatch in JSON payload", FailureCategory.DATA_ERROR),
        ("test is known flaky due to timing", FailureCategory.FLAKY),
        ("non-deterministic result in random seed", FailureCategory.FLAKY),
        ("race condition in concurrent writes", FailureCategory.FLAKY),
        ("completely opaque failure with no signal", FailureCategory.UNKNOWN),
    ],
)
def test_classify_category(text: str, expected_category: FailureCategory) -> None:
    result = _classify(text)
    assert result.category == expected_category  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Confidence is always ≤ 0.40
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "AssertionError: assert 1 == 2",
        "ModuleNotFoundError: No module named 'x'",
        "fixture 'x' not found error",
        "TimeoutError: timed out",
        "ConnectionError: refused",
        "KeyError: env var X",
        "FileNotFoundError: No such file",
        "flaky test",
        "completely unknown error",
    ],
)
def test_confidence_always_le_040(text: str) -> None:
    result = classify_by_heuristic(text, text, text)
    assert result.confidence <= 0.40


# ---------------------------------------------------------------------------
# is_flaky flag
# ---------------------------------------------------------------------------


def test_is_flaky_true_for_flaky_category() -> None:
    result = _classify("this test is known flaky due to timing issues")
    assert result.category == FailureCategory.FLAKY  # type: ignore[union-attr]
    assert result.is_flaky is True  # type: ignore[union-attr]


def test_is_flaky_false_for_non_flaky() -> None:
    result = _classify("AssertionError: assert 1 == 2")
    assert result.is_flaky is False  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Unknown fallback
# ---------------------------------------------------------------------------


def test_unknown_has_lowest_confidence() -> None:
    result = classify_by_heuristic("completely random error", "", "")
    assert result.category == FailureCategory.UNKNOWN
    assert result.confidence == 0.10


def test_unknown_fix_hint_empty() -> None:
    result = classify_by_heuristic("no pattern matches here", "", "")
    assert result.fix_hint == ""


# ---------------------------------------------------------------------------
# Rule priority (first match wins)
# ---------------------------------------------------------------------------


def test_assertion_error_takes_priority_over_unknown() -> None:
    # Both assertion and some random text — should match AssertionError first
    result = classify_by_heuristic("AssertionError", "assert x == y", "some traceback")
    assert result.category == FailureCategory.ASSERTION_ERROR


def test_classify_combines_all_three_args() -> None:
    # Only error_message contains the pattern
    result = classify_by_heuristic("UnknownError", "TimeoutError: timed out", "")
    assert result.category == FailureCategory.TIMEOUT


# ---------------------------------------------------------------------------
# Summary and explanation are non-empty for matched rules
# ---------------------------------------------------------------------------


def test_matched_rule_has_non_empty_summary() -> None:
    result = _classify("ModuleNotFoundError: No module named 'x'")
    assert result.summary  # type: ignore[union-attr]
    assert result.explanation  # type: ignore[union-attr]
    assert result.fix_hint  # type: ignore[union-attr]
