from __future__ import annotations

import time

import aiosqlite
from fastapi import APIRouter, HTTPException, Request

from analyzer.config import get_settings
from analyzer.llm.client import LLMClient
from analyzer.models.api import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
)
from analyzer.pipeline.ingestion import ingest_batch, ingest_log
from analyzer.pipeline.inference import run_inference, run_inference_batch
from analyzer.pipeline.scoring import score_result
from analyzer.pipeline.storage import store_batch, store_result
from analyzer.pipeline.validation import validate_result

router = APIRouter(tags=["analyze"])


def _to_response(record_id: str, scored: object) -> AnalyzeResponse:  # type: ignore[type-arg]
    from analyzer.models.pipeline import ScoredResult

    assert isinstance(scored, ScoredResult)
    h = scored.validation.inference.hypothesis
    return AnalyzeResponse(
        record_id=record_id,
        test_id=scored.validation.inference.parsed_failure.test_id,
        category=h.category,
        summary=h.summary,
        explanation=h.explanation,
        fix_hint=h.fix_hint,
        confidence=h.confidence,
        is_flaky=h.is_flaky,
        fallback_used=scored.validation.inference.fallback_used,
        fallback_source=scored.validation.inference.fallback_source,
        latency_ms=scored.validation.inference.latency_ms,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    settings = get_settings()
    db: aiosqlite.Connection = request.app.state.db
    client: LLMClient = request.app.state.llm

    try:
        failure = await ingest_log(body.raw_log, body.test_id)
        inference = await run_inference(failure, client, settings.confidence_threshold)
        validation = validate_result(inference, settings.confidence_threshold)
        scored = score_result(validation)
        record = await store_result(scored, db)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _to_response(record.record_id, scored)


@router.post("/batch-analyze", response_model=BatchAnalyzeResponse)
async def batch_analyze(body: BatchAnalyzeRequest, request: Request) -> BatchAnalyzeResponse:
    settings = get_settings()
    db: aiosqlite.Connection = request.app.state.db
    client: LLMClient = request.app.state.llm

    start = time.perf_counter()
    try:
        failures = await ingest_batch(body.logs, body.test_ids)
        inferences = await run_inference_batch(
            failures, client, settings.confidence_threshold, settings.max_concurrent_llm
        )
        validations = [validate_result(i, settings.confidence_threshold) for i in inferences]
        scored_list = [score_result(v) for v in validations]
        records = await store_batch(scored_list, db)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    total_ms = (time.perf_counter() - start) * 1000
    responses = [_to_response(r.record_id, s) for r, s in zip(records, scored_list)]

    return BatchAnalyzeResponse(
        results=responses,
        total=len(responses),
        latency_ms=round(total_ms, 2),
    )
