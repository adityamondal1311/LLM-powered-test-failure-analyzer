"""Integration tests for FastAPI routes — LLMClient mocked via AsyncMock."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from analyzer.api.app import create_app
from analyzer.config import Settings
from analyzer.llm.client import LLMClient
from analyzer.models.pipeline import FailureCategory, FallbackSource, RootCauseHypothesis
from analyzer.pipeline.storage import init_db
from tests.fixtures.sample_logs import ASSERTION_ERROR_LOG, IMPORT_ERROR_LOG
from tests.fixtures.sample_responses import LLM_ASSERTION, SAMPLE_USAGE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _llm_client_for(hypothesis: RootCauseHypothesis) -> LLMClient:
    client = MagicMock(spec=LLMClient)
    client._model = "claude-sonnet-4-6"
    client.classify_failure = AsyncMock(return_value=(hypothesis, SAMPLE_USAGE))
    return client


@pytest_asyncio.fixture
async def api_client(tmp_path: Any) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client backed by the FastAPI app with a mock LLM.

    ASGITransport does not fire lifespan events, so we set app.state directly.
    """
    db_path = str(tmp_path / "test.db")
    db = await aiosqlite.connect(db_path)
    await init_db(db)

    app = create_app()
    app.state.db = db
    app.state.llm = _llm_client_for(LLM_ASSERTION)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await db.close()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_returns_ok(api_client: AsyncClient) -> None:
    resp = await api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"] == "claude-sonnet-4-6"
    assert body["db"] == "connected"


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_empty_db(api_client: AsyncClient) -> None:
    resp = await api_client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_analyzed"] == 0
    assert body["avg_confidence"] == 0.0
    assert body["fallback_rate"] == 0.0


# ---------------------------------------------------------------------------
# POST /api/v1/analyze
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_returns_200(api_client: AsyncClient) -> None:
    resp = await api_client.post(
        "/api/v1/analyze",
        json={"raw_log": ASSERTION_ERROR_LOG},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_analyze_response_schema(api_client: AsyncClient) -> None:
    resp = await api_client.post(
        "/api/v1/analyze",
        json={"raw_log": ASSERTION_ERROR_LOG},
    )
    body = resp.json()
    assert "record_id" in body
    assert "category" in body
    assert "confidence" in body
    assert "fallback_used" in body
    assert "latency_ms" in body


@pytest.mark.asyncio
async def test_analyze_category_from_mock(api_client: AsyncClient) -> None:
    resp = await api_client.post(
        "/api/v1/analyze",
        json={"raw_log": ASSERTION_ERROR_LOG},
    )
    body = resp.json()
    assert body["category"] == "assertion_error"


@pytest.mark.asyncio
async def test_analyze_with_explicit_test_id(api_client: AsyncClient) -> None:
    resp = await api_client.post(
        "/api/v1/analyze",
        json={"raw_log": ASSERTION_ERROR_LOG, "test_id": "my_test_id"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["test_id"] == "my_test_id"


@pytest.mark.asyncio
async def test_analyze_rejects_empty_log(api_client: AsyncClient) -> None:
    resp = await api_client.post(
        "/api/v1/analyze",
        json={"raw_log": "short"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_analyze_metrics_updated_after_call(api_client: AsyncClient) -> None:
    await api_client.post("/api/v1/analyze", json={"raw_log": ASSERTION_ERROR_LOG})
    resp = await api_client.get("/metrics")
    body = resp.json()
    assert body["total_analyzed"] == 1


# ---------------------------------------------------------------------------
# POST /api/v1/batch-analyze
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_analyze_returns_all_results(api_client: AsyncClient) -> None:
    resp = await api_client.post(
        "/api/v1/batch-analyze",
        json={"logs": [ASSERTION_ERROR_LOG, IMPORT_ERROR_LOG]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert len(body["results"]) == 2


@pytest.mark.asyncio
async def test_batch_analyze_includes_latency(api_client: AsyncClient) -> None:
    resp = await api_client.post(
        "/api/v1/batch-analyze",
        json={"logs": [ASSERTION_ERROR_LOG]},
    )
    body = resp.json()
    assert body["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_batch_analyze_rejects_empty_list(api_client: AsyncClient) -> None:
    resp = await api_client.post(
        "/api/v1/batch-analyze",
        json={"logs": []},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/evaluate + GET /api/v1/evaluate/{job_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_returns_202_with_job_id(api_client: AsyncClient) -> None:
    resp = await api_client.post(
        "/api/v1/evaluate",
        json={"limit": 5},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["status"] in ("queued", "running", "completed", "failed")


@pytest.mark.asyncio
async def test_evaluate_job_polling(api_client: AsyncClient) -> None:
    resp = await api_client.post("/api/v1/evaluate", json={})
    job_id = resp.json()["job_id"]
    poll = await api_client.get(f"/api/v1/evaluate/{job_id}")
    assert poll.status_code == 200
    assert poll.json()["job_id"] == job_id


@pytest.mark.asyncio
async def test_evaluate_unknown_job_returns_404(api_client: AsyncClient) -> None:
    resp = await api_client.get("/api/v1/evaluate/doesnotexist")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Fallback path: API routes still succeed when LLM fails
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def api_client_rate_limited(tmp_path: Any) -> AsyncGenerator[AsyncClient, None]:
    import anthropic

    db_path = str(tmp_path / "test_rl.db")
    db = await aiosqlite.connect(db_path)
    await init_db(db)

    rl_client = MagicMock(spec=LLMClient)
    rl_client._model = "claude-sonnet-4-6"
    rl_client.classify_failure = AsyncMock(
        side_effect=anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
    )

    app = create_app()
    app.state.db = db
    app.state.llm = rl_client

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await db.close()


@pytest.mark.asyncio
async def test_analyze_falls_back_on_rate_limit(
    api_client_rate_limited: AsyncClient,
) -> None:
    resp = await api_client_rate_limited.post(
        "/api/v1/analyze",
        json={"raw_log": ASSERTION_ERROR_LOG},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["fallback_used"] is True
    assert body["confidence"] <= 0.40
