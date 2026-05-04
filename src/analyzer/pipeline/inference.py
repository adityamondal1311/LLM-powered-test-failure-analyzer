from __future__ import annotations

import asyncio
import time

import anthropic

from analyzer.fallback.heuristics import classify_by_heuristic
from analyzer.llm.client import LLMClient
from analyzer.models.pipeline import (
    FallbackSource,
    InferenceResult,
    ParsedFailure,
    RootCauseHypothesis,
)


async def run_inference(
    failure: ParsedFailure,
    client: LLMClient,
    confidence_threshold: float = 0.65,
) -> InferenceResult:
    """Run LLM inference on a single failure, falling back to heuristics on error."""
    start = time.perf_counter()
    fallback_used = False
    fallback_source = FallbackSource.LLM
    input_tokens = output_tokens = cache_hit_tokens = 0
    hypothesis: RootCauseHypothesis

    try:
        hypothesis, usage = await client.classify_failure(failure)
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_hit_tokens = usage.get("cache_hit_tokens", 0)

        # Route to fallback if LLM returned low confidence
        if hypothesis.confidence < confidence_threshold:
            tb_text = "\n".join(failure.traceback_lines)
            heuristic = classify_by_heuristic(
                failure.error_type, failure.error_message, tb_text
            )
            # Keep LLM hypothesis only if heuristic confidence is lower
            if heuristic.confidence > hypothesis.confidence:
                hypothesis = heuristic
                fallback_used = True
                fallback_source = FallbackSource.HEURISTIC

    except (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        asyncio.TimeoutError,
        ValueError,
    ):
        tb_text = "\n".join(failure.traceback_lines)
        hypothesis = classify_by_heuristic(
            failure.error_type, failure.error_message, tb_text
        )
        fallback_used = True
        fallback_source = FallbackSource.HEURISTIC

    latency_ms = (time.perf_counter() - start) * 1000

    return InferenceResult(
        parsed_failure=failure,
        hypothesis=hypothesis,
        model_id=client._model,
        latency_ms=round(latency_ms, 2),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit_tokens=cache_hit_tokens,
        fallback_used=fallback_used,
        fallback_source=fallback_source,
    )


async def run_inference_batch(
    failures: list[ParsedFailure],
    client: LLMClient,
    confidence_threshold: float = 0.65,
    max_concurrent: int = 5,
) -> list[InferenceResult]:
    """Run inference on a batch concurrently, bounded by a semaphore."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _bounded(f: ParsedFailure) -> InferenceResult:
        async with sem:
            return await run_inference(f, client, confidence_threshold)

    return list(await asyncio.gather(*[_bounded(f) for f in failures]))
