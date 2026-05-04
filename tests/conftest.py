"""Shared pytest fixtures for unit and integration tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from analyzer.api.app import create_app
from analyzer.config import Settings
from analyzer.llm.client import LLMClient
from analyzer.models.pipeline import (
    FailureCategory,
    FallbackSource,
    InferenceResult,
    ParsedFailure,
    RootCauseHypothesis,
    ScoredResult,
    ValidationResult,
)
from analyzer.pipeline.storage import init_db
from tests.fixtures.sample_responses import LLM_ASSERTION, SAMPLE_USAGE


# ---------------------------------------------------------------------------
# Core model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_hypothesis() -> RootCauseHypothesis:
    return LLM_ASSERTION


@pytest.fixture
def sample_parsed_failure() -> ParsedFailure:
    return ParsedFailure(
        test_id="tests/test_payments.py::test_user_balance",
        test_file="tests/test_payments.py",
        test_function="test_user_balance",
        error_type="AssertionError",
        error_message="assert 150.0 == 200.0",
        traceback_lines=[
            "_ test_user_balance _",
            "tests/test_payments.py:42: in test_user_balance",
            "    assert result == expected",
            "E   AssertionError: assert 150.0 == 200.0",
        ],
        duration_ms=230.0,
        raw_log="FAILED tests/test_payments.py::test_user_balance - AssertionError: assert 150.0 == 200.0",
        token_estimate=42,
    )


@pytest.fixture
def sample_inference_result(
    sample_parsed_failure: ParsedFailure,
    sample_hypothesis: RootCauseHypothesis,
) -> InferenceResult:
    return InferenceResult(
        parsed_failure=sample_parsed_failure,
        hypothesis=sample_hypothesis,
        model_id="claude-sonnet-4-6",
        latency_ms=450.0,
        input_tokens=150,
        output_tokens=80,
        cache_hit_tokens=120,
        fallback_used=False,
        fallback_source=FallbackSource.LLM,
    )


@pytest.fixture
def sample_validation_result(sample_inference_result: InferenceResult) -> ValidationResult:
    return ValidationResult(
        inference=sample_inference_result,
        schema_valid=True,
        confidence_passed=True,
        issues=[],
    )


@pytest.fixture
def sample_scored_result(sample_validation_result: ValidationResult) -> ScoredResult:
    h = sample_validation_result.inference.hypothesis
    return ScoredResult(
        validation=sample_validation_result,
        final_confidence=h.confidence,
        rank_score=0.95 * 0.6 + 0.2 + 0.2,
        actionable=True,
        routed_to=FallbackSource.LLM,
    )


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_client(sample_hypothesis: RootCauseHypothesis) -> LLMClient:
    """AsyncMock LLMClient that returns LLM_ASSERTION + SAMPLE_USAGE."""
    client = MagicMock(spec=LLMClient)
    client._model = "claude-sonnet-4-6"
    client.classify_failure = AsyncMock(return_value=(sample_hypothesis, SAMPLE_USAGE))
    return client


@pytest.fixture
def mock_llm_client_rate_limit() -> LLMClient:
    """LLMClient mock that raises RateLimitError on classify_failure."""
    import anthropic

    client = MagicMock(spec=LLMClient)
    client._model = "claude-sonnet-4-6"
    client.classify_failure = AsyncMock(
        side_effect=anthropic.RateLimitError(
            message="rate limit",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
    )
    return client


@pytest.fixture
def mock_llm_client_timeout() -> LLMClient:
    """LLMClient mock that raises asyncio.TimeoutError."""
    client = MagicMock(spec=LLMClient)
    client._model = "claude-sonnet-4-6"
    client.classify_failure = AsyncMock(side_effect=asyncio.TimeoutError())
    return client


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def tmp_db(tmp_path: Any) -> AsyncGenerator[aiosqlite.Connection, None]:
    """Temporary aiosqlite database, initialized with the schema."""
    db_file = tmp_path / "test.db"
    async with aiosqlite.connect(str(db_file)) as db:
        await init_db(db)
        yield db


# ---------------------------------------------------------------------------
# App / HTTP client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings_override(tmp_path: Any) -> Settings:
    """Settings pointing to a temporary db path with a dummy API key."""
    return Settings(
        anthropic_api_key="sk-ant-dummy-key",
        db_path=str(tmp_path / "test.db"),
        model_id="claude-sonnet-4-6",
        llm_timeout_ms=2000,
        llm_max_retries=1,
        confidence_threshold=0.65,
    )


@pytest.fixture
def test_app(
    settings_override: Settings,
    mock_llm_client: LLMClient,
    tmp_path: Any,
) -> Generator[TestClient, None, None]:
    """Sync TestClient with mock LLMClient injected into app state."""
    app = create_app()

    async def _override_lifespan(app: Any) -> AsyncGenerator[None, None]:
        db = await aiosqlite.connect(settings_override.db_path)
        await init_db(db)
        app.state.db = db
        app.state.llm = mock_llm_client
        yield
        await db.close()

    # Patch the lifespan for this test
    from contextlib import asynccontextmanager

    app.router.lifespan_context = asynccontextmanager(_override_lifespan)

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


@pytest_asyncio.fixture
async def async_test_client(
    settings_override: Settings,
    mock_llm_client: LLMClient,
) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTPX client with mock LLMClient — for async integration tests."""
    app = create_app()

    async def _override_lifespan(app: Any) -> AsyncGenerator[None, None]:
        db = await aiosqlite.connect(settings_override.db_path)
        await init_db(db)
        app.state.db = db
        app.state.llm = mock_llm_client
        yield
        await db.close()

    from contextlib import asynccontextmanager

    app.router.lifespan_context = asynccontextmanager(_override_lifespan)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
