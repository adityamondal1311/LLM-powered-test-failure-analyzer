"""Integration tests: full pipeline from raw log to StoredRecord, no real LLM calls."""

from __future__ import annotations

import pytest

from analyzer.models.pipeline import FailureCategory, FallbackSource
from analyzer.pipeline.ingestion import ingest_log
from analyzer.pipeline.inference import run_inference
from analyzer.pipeline.scoring import score_result
from analyzer.pipeline.storage import store_result
from analyzer.pipeline.validation import validate_result
from tests.fixtures.sample_logs import (
    ASSERTION_ERROR_LOG,
    IMPORT_ERROR_LOG,
    NETWORK_ERROR_LOG,
    TIMEOUT_LOG,
)
from tests.fixtures.sample_responses import LLM_ASSERTION, SAMPLE_USAGE


# ---------------------------------------------------------------------------
# Full pipeline happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_happy_path(
    mock_llm_client: object,
    tmp_db: object,
) -> None:
    failure = await ingest_log(ASSERTION_ERROR_LOG)
    inference = await run_inference(failure, mock_llm_client)  # type: ignore[arg-type]
    validation = validate_result(inference)
    scored = score_result(validation)
    record = await store_result(scored, tmp_db)  # type: ignore[arg-type]

    assert record.record_id
    assert scored.validation.inference.hypothesis.category == FailureCategory.ASSERTION_ERROR
    assert scored.routed_to == FallbackSource.LLM
    assert scored.actionable is True
    assert record.scored_result is scored


@pytest.mark.asyncio
async def test_full_pipeline_produces_valid_record_id(
    mock_llm_client: object,
    tmp_db: object,
) -> None:
    import uuid

    failure = await ingest_log(ASSERTION_ERROR_LOG)
    inference = await run_inference(failure, mock_llm_client)  # type: ignore[arg-type]
    validation = validate_result(inference)
    scored = score_result(validation)
    record = await store_result(scored, tmp_db)  # type: ignore[arg-type]
    uuid.UUID(record.record_id)  # raises ValueError if not a valid UUID


@pytest.mark.asyncio
async def test_full_pipeline_import_error_log(
    mock_llm_client: object,
    tmp_db: object,
) -> None:
    from tests.fixtures.sample_responses import LLM_IMPORT
    from unittest.mock import AsyncMock, MagicMock
    from analyzer.llm.client import LLMClient

    client = MagicMock(spec=LLMClient)
    client._model = "claude-sonnet-4-6"
    client.classify_failure = AsyncMock(return_value=(LLM_IMPORT, SAMPLE_USAGE))

    failure = await ingest_log(IMPORT_ERROR_LOG)
    inference = await run_inference(failure, client)  # type: ignore[arg-type]
    validation = validate_result(inference)
    scored = score_result(validation)
    record = await store_result(scored, tmp_db)  # type: ignore[arg-type]

    assert scored.validation.inference.hypothesis.category == FailureCategory.IMPORT_ERROR
    assert record.record_id


# ---------------------------------------------------------------------------
# Full pipeline fallback path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_fallback_on_api_error(
    mock_llm_client_rate_limit: object,
    tmp_db: object,
) -> None:
    failure = await ingest_log(ASSERTION_ERROR_LOG)
    inference = await run_inference(failure, mock_llm_client_rate_limit)  # type: ignore[arg-type]
    validation = validate_result(inference)
    scored = score_result(validation)
    record = await store_result(scored, tmp_db)  # type: ignore[arg-type]

    assert inference.fallback_used is True
    assert inference.hypothesis.confidence <= 0.40
    assert scored.routed_to == FallbackSource.HEURISTIC
    assert record.record_id


@pytest.mark.asyncio
async def test_full_pipeline_fallback_timeout(
    mock_llm_client_timeout: object,
    tmp_db: object,
) -> None:
    failure = await ingest_log(TIMEOUT_LOG)
    inference = await run_inference(failure, mock_llm_client_timeout)  # type: ignore[arg-type]
    assert inference.fallback_used is True
    assert inference.fallback_source == FallbackSource.HEURISTIC
    # Heuristic should detect TimeoutError
    assert inference.hypothesis.category == FailureCategory.TIMEOUT


# ---------------------------------------------------------------------------
# Multiple logs through the pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_multiple_logs_stored(
    mock_llm_client: object,
    tmp_db: object,
) -> None:
    logs = [ASSERTION_ERROR_LOG, IMPORT_ERROR_LOG, NETWORK_ERROR_LOG]
    records = []
    for log in logs:
        failure = await ingest_log(log)
        inference = await run_inference(failure, mock_llm_client)  # type: ignore[arg-type]
        validation = validate_result(inference)
        scored = score_result(validation)
        records.append(await store_result(scored, tmp_db))  # type: ignore[arg-type]

    assert len(records) == 3
    # All record IDs are unique
    ids = {r.record_id for r in records}
    assert len(ids) == 3


# ---------------------------------------------------------------------------
# Validation confidence gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_confidence_does_not_pass_gate(mock_llm_client: object) -> None:
    from analyzer.models.pipeline import RootCauseHypothesis
    from unittest.mock import AsyncMock

    low_conf = RootCauseHypothesis(
        category=FailureCategory.UNKNOWN,
        summary="Unsure about this one",
        explanation="Not enough information to classify.",
        fix_hint="",
        confidence=0.30,
        is_flaky=False,
    )
    mock_llm_client.classify_failure = AsyncMock(return_value=(low_conf, SAMPLE_USAGE))  # type: ignore[union-attr]

    failure = await ingest_log(ASSERTION_ERROR_LOG)
    inference = await run_inference(failure, mock_llm_client)  # type: ignore[arg-type]
    validation = validate_result(inference, threshold=0.65)

    # The heuristic kicks in (confidence 0.40 > 0.30), so inference shows fallback
    # But validation still checks whatever hypothesis ended up being used
    assert validation.confidence_passed == (inference.hypothesis.confidence >= 0.65)
