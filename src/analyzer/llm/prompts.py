from __future__ import annotations

from analyzer.models.pipeline import ParsedFailure

SYSTEM_PROMPT = """\
You are an expert Python test failure analyst. Your task is to classify pytest \
test failures and identify root causes with high precision.

You will be given a pytest failure log and must call the `classify_test_failure` \
tool with a structured analysis.

## Failure Categories
- assertion_error: Value comparison failures, unexpected output, failed assertions
- import_error: Missing modules, circular imports, wrong package versions
- fixture_error: pytest fixture setup/teardown failures, scope issues, missing fixtures
- timeout: Test exceeded time limit or hung on I/O
- environment_error: Missing env vars, wrong Python version, OS-level issues
- network_error: External service unreachable, DNS failure, SSL errors
- data_error: Corrupt test data, schema mismatch, missing fixture files
- flaky: Non-deterministic failure (race conditions, timing, random seeds)
- unknown: Cannot be classified with reasonable confidence

## Confidence Calibration
- 0.90–1.00: Unambiguous error with clear traceback pointing to root cause
- 0.70–0.89: Strong signal but some ambiguity (e.g. wrapped exceptions)
- 0.50–0.69: Partial information; category likely but not certain
- 0.00–0.49: Insufficient information; prefer unknown category

## fix_hint Guidelines
- Be specific and actionable (e.g. "Check that REDIS_URL env var is set in CI")
- Never suggest "run the tests again" — always name a code or config change
- If truly unknown, set fix_hint to empty string ""
"""


def build_user_message(failure: ParsedFailure) -> str:
    """Build the per-request user message.
    All volatile content lives here so the cached system prefix is never invalidated.
    """
    traceback_text = "\n".join(failure.traceback_lines[-40:])
    return (
        f"## Test Failure to Classify\n\n"
        f"**Test:** `{failure.test_id}`\n"
        f"**File:** `{failure.test_file}`\n"
        f"**Duration:** {failure.duration_ms:.0f}ms\n"
        f"**Error Type:** `{failure.error_type}`\n\n"
        f"**Error Message:**\n```\n{failure.error_message[:600]}\n```\n\n"
        f"**Traceback (tail):**\n```\n{traceback_text[:2000]}\n```\n\n"
        "Call `classify_test_failure` with your analysis."
    )
