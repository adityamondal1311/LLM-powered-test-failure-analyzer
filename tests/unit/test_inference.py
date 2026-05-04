"""Unit tests for pipeline/inference.py — LLMClient fully mocked."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest

from analyzer.models.pipeline import (
    FailureCategory,
    FallbackSource,
    RootCauseHypothesis,
)
from analyzer.pipeline.inference import run_inference, run_inference_batch
from tests.fixtures.sample_responses import (
    HEURISTIC_ASSERTION,
    LOW_CONFIDENCE,
    LLM_ASSERTION,
    SAMPLE_USAGE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    hypothesis: RootCauseHypothesis,
    usage: dict = SAMPLE_USAGE,
) -> object:
    client = MagicMock()
    client._model = "claude-sonnet-4-6"
    client.classify_failure = AsyncMock(return_value=(hypothesis, usage))
    return client


def _make_failing_client(exc: Exception) -> object:
    client = MagicMock()
    client._model = "claude-sonnet-4-6"
    client.classify_failure = AsyncMock(side_effect=exc)
    return client


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_no_fallback(sample_parsed_failure: object) -> None:
    client = _make_client(LLM_ASSERTION)
    result = await run_inference(sample_parsed_failure, client)  # type: ignore[arg-type]
    assert result.fallback_used is False
    assert result.fallback_source == FallbackSource.LLM
    assert result.hypothesis.category == FailureCategory.ASSERTION_ERROR
    assert result.hypothesis.confidence == 0.95


@pytest.mark.asyncio
async def test_happy_path_usage_populated(sample_parsed_failure: object) -> None:
    client = _make_client(LLM_ASSERTION, SAMPLE_USAGE)
    result = await run_inference(sample_parsed_failure, client)  # type: ignore[arg-type]
    assert result.input_tokens == SAMPLE_USAGE["input_tokens"]
    assert result.output_tokens == SAMPLE_USAGE["output_tokens"]
    assert result.cache_hit_tokens == SAMPLE_USAGE["cache_hit_tokens"]


@pytest.mark.asyncio
async def test_happy_path_model_id(sample_parsed_failure: object) -> None:
    client = _make_client(LLM_ASSERTION)
    result = await run_inference(sample_parsed_failure, client)  # type: ignore[arg-type]
    assert result.model_id == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_latency_ms_is_positive(sample_parsed_failure: object) -> None:
    client = _make_client(LLM_ASSERTION)
    result = await run_inference(sample_parsed_failure, client)  # type: ignore[arg-type]
    assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# Fallback on API errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_on_rate_limit(sample_parsed_failure: object) -> None:
    exc = anthropic.RateLimitError(
        message="rate limited",
        response=MagicMock(status_code=429, headers={}),
        body={},
    )
    client = _make_failing_client(exc)
    result = await run_inference(sample_parsed_failure, client)  # type: ignore[arg-type]
    assert result.fallback_used is True
    assert result.fallback_source == FallbackSource.HEURISTIC
    assert result.hypothesis.confidence <= 0.40


@pytest.mark.asyncio
async def test_fallback_on_internal_server_error(sample_parsed_failure: object) -> None:
    exc = anthropic.InternalServerError(
        message="server error",
        response=MagicMock(status_code=500, headers={}),
        body={},
    )
    client = _make_failing_client(exc)
    result = await run_inference(sample_parsed_failure, client)  # type: ignore[arg-type]
    assert result.fallback_used is True
    assert result.hypothesis.confidence <= 0.40


@pytest.mark.asyncio
async def test_fallback_on_timeout(sample_parsed_failure: object) -> None:
    client = _make_failing_client(asyncio.TimeoutError())
    result = await run_inference(sample_parsed_failure, client)  # type: ignore[arg-type]
    assert result.fallback_used is True
    assert result.fallback_source == FallbackSource.HEURISTIC


@pytest.mark.asyncio
async def test_fallback_on_value_error(sample_parsed_failure: object) -> None:
    client = _make_failing_client(ValueError("No tool_use block in LLM response"))
    result = await run_inference(sample_parsed_failure, client)  # type: ignore[arg-type]
    assert result.fallback_used is True


# ---------------------------------------------------------------------------
# Low-confidence routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_confidence_llm_routes_to_heuristic_if_better(
    sample_parsed_failure: object,
) -> None:
    """When LLM returns low confidence, fall back if heuristic is better."""
    client = _make_client(LOW_CONFIDENCE)
    result = await run_inference(sample_parsed_failure, client)  # type: ignore[arg-type]
    # sample_parsed_failure has AssertionError which heuristic matches at 0.40 > 0.30
    assert result.fallback_used is True
    assert result.hypothesis.confidence == 0.40


@pytest.mark.asyncio
async def test_low_confidence_kept_if_heuristic_is_worse(
    sample_parsed_failure: object,
) -> None:
    """If LLM confidence < threshold but > heuristic confidence, keep LLM result."""
    # A failure log that heuristic cannot match → heuristic returns 0.10 (unknown)
    from analyzer.models.pipeline import ParsedFailure

    opaque_failure = ParsedFailure(
        test_id="tests/test_x.py::test_opaque",
        test_file="tests/test_x.py",
        test_function="test_opaque",
        error_type="RecursionError",
        error_message="maximum recursion depth exceeded",
        traceback_lines=["RecursionError: maximum recursion depth exceeded"],
        duration_ms=0.0,
        raw_log="FAILED tests/test_x.py::test_opaque - RecursionError",
        token_estimate=10,
    )
    mid_conf = RootCauseHypothesis(
        category=FailureCategory.UNKNOWN,
        summary="LLM guessed unknown",
        explanation="Hard to tell",
        fix_hint="check manually",
        confidence=0.20,  # below threshold (0.65) but above heuristic floor (0.10)
        is_flaky=False,
    )
    client = _make_client(mid_conf)
    result = await run_inference(opaque_failure, client)
    # heuristic gives 0.10, LLM gave 0.20 → keep LLM result
    assert result.fallback_used is False
    assert result.hypothesis.confidence == 0.20


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_returns_all_results(sample_parsed_failure: object) -> None:
    client = _make_client(LLM_ASSERTION)
    failures = [sample_parsed_failure] * 5  # type: ignore[list-item]
    results = await run_inference_batch(failures, client)  # type: ignore[arg-type]
    assert len(results) == 5


@pytest.mark.asyncio
async def test_batch_semaphore_limits_concurrency(sample_parsed_failure: object) -> None:
    call_count = 0
    concurrent_peak = 0
    active = 0

    async def _slow_classify(failure: object) -> tuple:
        nonlocal call_count, concurrent_peak, active
        active += 1
        concurrent_peak = max(concurrent_peak, active)
        await asyncio.sleep(0)
        call_count += 1
        active -= 1
        return (LLM_ASSERTION, SAMPLE_USAGE)

    client = MagicMock()
    client._model = "claude-sonnet-4-6"
    client.classify_failure = _slow_classify

    failures = [sample_parsed_failure] * 10  # type: ignore[list-item]
    await run_inference_batch(failures, client, max_concurrent=3)  # type: ignore[arg-type]
    assert call_count == 10
    assert concurrent_peak <= 3
