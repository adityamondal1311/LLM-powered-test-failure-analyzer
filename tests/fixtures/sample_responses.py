"""Sample RootCauseHypothesis objects for use in unit tests."""

from __future__ import annotations

from analyzer.models.pipeline import FailureCategory, RootCauseHypothesis

# High-confidence LLM-style responses (confidence > 0.65)
LLM_ASSERTION = RootCauseHypothesis(
    category=FailureCategory.ASSERTION_ERROR,
    summary="Value comparison failed: get_balance returned 150.0 instead of 200.0",
    explanation=(
        "The test asserts that get_balance returns 200.0 for user_id=7, "
        "but the function returned 150.0. This indicates either a logic bug "
        "in the balance calculation or stale test data."
    ),
    fix_hint="Review get_balance logic for user_id=7 and verify the test fixture creates the correct balance.",
    confidence=0.95,
    is_flaky=False,
)

LLM_IMPORT = RootCauseHypothesis(
    category=FailureCategory.IMPORT_ERROR,
    summary="Missing dependency: redis module not installed",
    explanation="Python cannot find the 'redis' package. It is likely a missing dev dependency.",
    fix_hint="Run `pip install redis` or add redis to pyproject.toml dev dependencies and reinstall.",
    confidence=0.98,
    is_flaky=False,
)

LLM_FIXTURE = RootCauseHypothesis(
    category=FailureCategory.FIXTURE_ERROR,
    summary="Fixture 'db_session' not found during test setup",
    explanation="pytest cannot locate the 'db_session' fixture. It may be missing from conftest.py.",
    fix_hint="Add db_session fixture to tests/conftest.py and ensure conftest.py is in the correct directory.",
    confidence=0.92,
    is_flaky=False,
)

LLM_TIMEOUT = RootCauseHypothesis(
    category=FailureCategory.TIMEOUT,
    summary="Database query exceeded 5 second timeout",
    explanation="The test_slow_query test timed out after 5 seconds, suggesting a missing index or blocking query.",
    fix_hint="Add a database index on the queried column or mock the database call in unit tests.",
    confidence=0.88,
    is_flaky=False,
)

LLM_NETWORK = RootCauseHypothesis(
    category=FailureCategory.NETWORK_ERROR,
    summary="HTTP connection to api.rates.io failed with max retries exceeded",
    explanation="The test makes a real network call that failed. External services should be mocked in tests.",
    fix_hint="Mock the requests.get call with unittest.mock or responses library to avoid real network calls.",
    confidence=0.93,
    is_flaky=False,
)

LLM_ENVIRONMENT = RootCauseHypothesis(
    category=FailureCategory.ENVIRONMENT_ERROR,
    summary="Missing environment variable SMTP_HOST",
    explanation="The test reads os.environ['SMTP_HOST'] which is not set in the test environment.",
    fix_hint="Add SMTP_HOST to .env.example and set it in the CI environment or use monkeypatch in tests.",
    confidence=0.91,
    is_flaky=False,
)

LLM_DATA = RootCauseHypothesis(
    category=FailureCategory.DATA_ERROR,
    summary="Test data file data/config.json not found",
    explanation="The test attempts to load a config file that does not exist at the expected path.",
    fix_hint="Create tests/fixtures/config.json or update the path in the test to point to an existing file.",
    confidence=0.96,
    is_flaky=False,
)

LLM_FLAKY = RootCauseHypothesis(
    category=FailureCategory.FLAKY,
    summary="Race condition in concurrent writer pool causes intermittent count mismatch",
    explanation="The test asserts 100 results but gets 99 due to a race condition in the writer pool.",
    fix_hint="Add proper locking around the writer pool or use pytest-xdist isolation to prevent shared state.",
    confidence=0.78,
    is_flaky=True,
)

LLM_UNKNOWN = RootCauseHypothesis(
    category=FailureCategory.UNKNOWN,
    summary="RecursionError in processor.run() — root cause unclear",
    explanation="Maximum recursion depth exceeded. Could be an infinite loop in processor logic.",
    fix_hint="Add a recursion depth guard or review processor.run() for unbounded recursive calls.",
    confidence=0.70,
    is_flaky=False,
)

# Low-confidence response (triggers fallback routing)
LOW_CONFIDENCE = RootCauseHypothesis(
    category=FailureCategory.UNKNOWN,
    summary="Unable to determine root cause from log",
    explanation="Insufficient information in the provided log to classify with confidence.",
    fix_hint="",
    confidence=0.30,
    is_flaky=False,
)

# Heuristic-style response (confidence <= 0.40)
HEURISTIC_ASSERTION = RootCauseHypothesis(
    category=FailureCategory.ASSERTION_ERROR,
    summary="[HEURISTIC] Matched pattern for assertion_error",
    explanation="Regex heuristic detected a assertion_error pattern in the log.",
    fix_hint="Review the assertion logic and compare expected vs actual values in the test.",
    confidence=0.40,
    is_flaky=False,
)

HEURISTIC_UNKNOWN = RootCauseHypothesis(
    category=FailureCategory.UNKNOWN,
    summary="[HEURISTIC] No matching pattern found.",
    explanation="No deterministic rule matched; manual inspection is required.",
    fix_hint="",
    confidence=0.10,
    is_flaky=False,
)

# Usage dict that classify_failure returns alongside the hypothesis
SAMPLE_USAGE: dict[str, int] = {
    "input_tokens": 150,
    "output_tokens": 80,
    "cache_hit_tokens": 120,
    "cache_write_tokens": 0,
    "latency_ms": 420,
}
