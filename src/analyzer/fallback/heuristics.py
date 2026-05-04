from __future__ import annotations

import re

from analyzer.models.pipeline import FailureCategory, RootCauseHypothesis

_RULES: list[tuple[re.Pattern[str], FailureCategory, str]] = [
    (
        re.compile(r"\bAssertionError\b|\bassert\b.+==", re.I),
        FailureCategory.ASSERTION_ERROR,
        "Review the assertion logic and compare expected vs actual values in the test.",
    ),
    (
        re.compile(r"\bModuleNotFoundError\b|\bImportError\b", re.I),
        FailureCategory.IMPORT_ERROR,
        "Check that all dependencies are installed: run `pip install -e '.[dev]'`.",
    ),
    (
        re.compile(r"\bfixture\b.+(?:not found|error|failed)", re.I),
        FailureCategory.FIXTURE_ERROR,
        "Verify fixture scope and that conftest.py is discoverable by pytest.",
    ),
    (
        re.compile(r"\bTimeoutError\b|\btimed?\s*out\b", re.I),
        FailureCategory.TIMEOUT,
        "Increase the test timeout or check for blocking I/O calls.",
    ),
    (
        re.compile(
            r"\bConnectionError\b|\bSSLError\b|\bDNS\b|\bHTTPError\b|\bConnectionRefused\b",
            re.I,
        ),
        FailureCategory.NETWORK_ERROR,
        "Check network connectivity and mock external services in unit tests.",
    ),
    (
        re.compile(
            r"\bEnvironmentError\b|\bKeyError\b.+env|\bos\.environ\b|\benv var\b", re.I
        ),
        FailureCategory.ENVIRONMENT_ERROR,
        "Ensure required environment variables are set; check .env.example.",
    ),
    (
        re.compile(r"\bFileNotFoundError\b|\bNo such file\b|\bschema mismatch\b", re.I),
        FailureCategory.DATA_ERROR,
        "Verify that test data files exist and match the expected schema.",
    ),
    (
        re.compile(r"\bflak[yi]\b|\bnon.deterministic\b|\brace condition\b", re.I),
        FailureCategory.FLAKY,
        "Mark the test with @pytest.mark.flaky and investigate timing/ordering issues.",
    ),
]


def classify_by_heuristic(
    error_type: str,
    error_message: str,
    traceback_text: str,
) -> RootCauseHypothesis:
    """Deterministic regex fallback. Returns confidence ≤ 0.40."""
    combined = f"{error_type}\n{error_message}\n{traceback_text}"

    for pattern, category, fix_hint in _RULES:
        if pattern.search(combined):
            return RootCauseHypothesis(
                category=category,
                summary=f"[HEURISTIC] Matched pattern for {category.value}",
                explanation=f"Regex heuristic detected a {category.value} pattern in the log.",
                fix_hint=fix_hint,
                confidence=0.40,
                is_flaky=(category == FailureCategory.FLAKY),
            )

    return RootCauseHypothesis(
        category=FailureCategory.UNKNOWN,
        summary="[HEURISTIC] No matching pattern found.",
        explanation="No deterministic rule matched; manual inspection is required.",
        fix_hint="",
        confidence=0.10,
        is_flaky=False,
    )
