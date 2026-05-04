from __future__ import annotations

import aiosqlite
from fastapi import APIRouter, Request

from analyzer.config import get_settings
from analyzer.models.api import HealthResponse, MetricsResponse
from analyzer.pipeline.storage import get_aggregate_stats

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    db: aiosqlite.Connection = request.app.state.db
    try:
        await db.execute("SELECT 1")
        db_status = "connected"
    except Exception:
        db_status = "error"

    return HealthResponse(
        status="ok",
        model=get_settings().model_id,
        db=db_status,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(request: Request) -> MetricsResponse:
    db: aiosqlite.Connection = request.app.state.db
    stats = await get_aggregate_stats(db)
    return MetricsResponse(
        total_analyzed=stats["total"],
        category_distribution=stats["category_distribution"],
        avg_confidence=stats["avg_confidence"],
        cache_hit_rate=0.0,  # populated by instrumentation layer in production
        fallback_rate=stats["fallback_rate"],
    )
