"""Unit tests for pipeline/storage.py."""

from __future__ import annotations

import pytest
import pytest_asyncio

from analyzer.models.pipeline import (
    FailureCategory,
    FallbackSource,
    InferenceResult,
    RootCauseHypothesis,
    ScoredResult,
    ValidationResult,
)
from analyzer.pipeline.storage import (
    get_aggregate_stats,
    init_db,
    query_by_category,
    store_batch,
    store_result,
)


def _make_scored(
    sample_parsed_failure: object,
    *,
    category: FailureCategory = FailureCategory.ASSERTION_ERROR,
    confidence: float = 0.90,
    fix_hint: str = "Check the assertion and expected values now",
    fallback_used: bool = False,
    fallback_source: FallbackSource = FallbackSource.LLM,
) -> ScoredResult:
    h = RootCauseHypothesis(
        category=category,
        summary="Test summary",
        explanation="Test explanation for this failure.",
        fix_hint=fix_hint,
        confidence=confidence,
        is_flaky=False,
    )
    inference = InferenceResult(
        parsed_failure=sample_parsed_failure,  # type: ignore[arg-type]
        hypothesis=h,
        model_id="claude-sonnet-4-6",
        latency_ms=300.0,
        input_tokens=100,
        output_tokens=40,
        cache_hit_tokens=80,
        fallback_used=fallback_used,
        fallback_source=fallback_source,
    )
    validation = ValidationResult(
        inference=inference,
        schema_valid=True,
        confidence_passed=True,
        issues=[],
    )
    rank = confidence * 0.6 + 0.2 + 0.2
    return ScoredResult(
        validation=validation,
        final_confidence=confidence,
        rank_score=round(rank, 4),
        actionable=True,
        routed_to=fallback_source if fallback_used else FallbackSource.LLM,
    )


# ---------------------------------------------------------------------------
# store_result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_result_returns_stored_record(
    tmp_db: object, sample_parsed_failure: object
) -> None:
    scored = _make_scored(sample_parsed_failure)
    record = await store_result(scored, tmp_db)  # type: ignore[arg-type]
    assert record.record_id
    assert record.stored_at_utc > 0
    assert record.scored_result is scored


@pytest.mark.asyncio
async def test_store_result_record_id_is_uuid(
    tmp_db: object, sample_parsed_failure: object
) -> None:
    scored = _make_scored(sample_parsed_failure)
    record = await store_result(scored, tmp_db)  # type: ignore[arg-type]
    import uuid

    uuid.UUID(record.record_id)  # raises if not a valid UUID


@pytest.mark.asyncio
async def test_store_result_unique_ids(tmp_db: object, sample_parsed_failure: object) -> None:
    scored = _make_scored(sample_parsed_failure)
    r1 = await store_result(scored, tmp_db)  # type: ignore[arg-type]
    r2 = await store_result(scored, tmp_db)  # type: ignore[arg-type]
    assert r1.record_id != r2.record_id


# ---------------------------------------------------------------------------
# query_by_category
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_by_category_returns_stored(
    tmp_db: object, sample_parsed_failure: object
) -> None:
    scored = _make_scored(sample_parsed_failure, category=FailureCategory.IMPORT_ERROR)
    await store_result(scored, tmp_db)  # type: ignore[arg-type]
    rows = await query_by_category("import_error", tmp_db)  # type: ignore[arg-type]
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_query_by_category_filters_correctly(
    tmp_db: object, sample_parsed_failure: object
) -> None:
    await store_result(
        _make_scored(sample_parsed_failure, category=FailureCategory.ASSERTION_ERROR),
        tmp_db,  # type: ignore[arg-type]
    )
    await store_result(
        _make_scored(sample_parsed_failure, category=FailureCategory.TIMEOUT),
        tmp_db,  # type: ignore[arg-type]
    )
    rows = await query_by_category("assertion_error", tmp_db)  # type: ignore[arg-type]
    assert len(rows) == 1
    rows_timeout = await query_by_category("timeout", tmp_db)  # type: ignore[arg-type]
    assert len(rows_timeout) == 1


@pytest.mark.asyncio
async def test_query_by_category_respects_limit(
    tmp_db: object, sample_parsed_failure: object
) -> None:
    for _ in range(5):
        await store_result(
            _make_scored(sample_parsed_failure, category=FailureCategory.NETWORK_ERROR),
            tmp_db,  # type: ignore[arg-type]
        )
    rows = await query_by_category("network_error", tmp_db, limit=3)  # type: ignore[arg-type]
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# store_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_batch_returns_all_records(
    tmp_db: object, sample_parsed_failure: object
) -> None:
    scored_list = [
        _make_scored(sample_parsed_failure, category=FailureCategory.ASSERTION_ERROR),
        _make_scored(sample_parsed_failure, category=FailureCategory.IMPORT_ERROR),
        _make_scored(sample_parsed_failure, category=FailureCategory.TIMEOUT),
    ]
    records = await store_batch(scored_list, tmp_db)  # type: ignore[arg-type]
    assert len(records) == 3


# ---------------------------------------------------------------------------
# get_aggregate_stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_aggregate_stats_empty_db(tmp_db: object) -> None:
    stats = await get_aggregate_stats(tmp_db)  # type: ignore[arg-type]
    assert stats["total"] == 0
    assert stats["avg_confidence"] == 0.0
    assert stats["fallback_rate"] == 0.0


@pytest.mark.asyncio
async def test_get_aggregate_stats_counts(tmp_db: object, sample_parsed_failure: object) -> None:
    await store_result(
        _make_scored(sample_parsed_failure, confidence=0.90, fallback_used=False),
        tmp_db,  # type: ignore[arg-type]
    )
    await store_result(
        _make_scored(
            sample_parsed_failure,
            confidence=0.40,
            fallback_used=True,
            fallback_source=FallbackSource.HEURISTIC,
        ),
        tmp_db,  # type: ignore[arg-type]
    )
    stats = await get_aggregate_stats(tmp_db)  # type: ignore[arg-type]
    assert stats["total"] == 2
    assert stats["avg_confidence"] == pytest.approx(0.65)


@pytest.mark.asyncio
async def test_get_aggregate_stats_category_distribution(
    tmp_db: object, sample_parsed_failure: object
) -> None:
    await store_result(
        _make_scored(sample_parsed_failure, category=FailureCategory.ASSERTION_ERROR),
        tmp_db,  # type: ignore[arg-type]
    )
    await store_result(
        _make_scored(sample_parsed_failure, category=FailureCategory.ASSERTION_ERROR),
        tmp_db,  # type: ignore[arg-type]
    )
    await store_result(
        _make_scored(sample_parsed_failure, category=FailureCategory.TIMEOUT),
        tmp_db,  # type: ignore[arg-type]
    )
    stats = await get_aggregate_stats(tmp_db)  # type: ignore[arg-type]
    dist = stats["category_distribution"]
    assert dist.get("assertion_error") == 2
    assert dist.get("timeout") == 1
