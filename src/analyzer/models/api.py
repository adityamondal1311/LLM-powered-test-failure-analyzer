from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from analyzer.models.pipeline import FailureCategory, FallbackSource


class AnalyzeRequest(BaseModel):
    raw_log: str = Field(..., min_length=10, max_length=50_000)
    test_id: str | None = None


class AnalyzeResponse(BaseModel):
    record_id: str
    test_id: str
    category: FailureCategory
    summary: str
    explanation: str
    fix_hint: str
    confidence: float
    is_flaky: bool
    fallback_used: bool
    fallback_source: FallbackSource
    latency_ms: float


class BatchAnalyzeRequest(BaseModel):
    logs: list[str] = Field(..., min_length=1, max_length=50)
    test_ids: list[str] | None = None


class BatchAnalyzeResponse(BaseModel):
    results: list[AnalyzeResponse]
    total: int
    latency_ms: float


class EvaluateRequest(BaseModel):
    dataset_path: str | None = None
    limit: int | None = Field(None, ge=1, le=200)


class EvaluateResponse(BaseModel):
    job_id: str
    status: str
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    accuracy: float | None = None
    n_samples: int | None = None
    details: list[dict[str, Any]] = []


class HealthResponse(BaseModel):
    status: str
    model: str
    db: str


class MetricsResponse(BaseModel):
    total_analyzed: int
    category_distribution: dict[str, int]
    avg_confidence: float
    cache_hit_rate: float
    fallback_rate: float
