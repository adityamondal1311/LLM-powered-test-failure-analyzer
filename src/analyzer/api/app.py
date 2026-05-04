from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any

import aiosqlite
from fastapi import FastAPI

from analyzer.api.middleware import TimingMiddleware
from analyzer.api.routes import analyze, evaluate, health
from analyzer.config import get_settings
from analyzer.llm.client import LLMClient
from analyzer.pipeline.storage import init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    db = await aiosqlite.connect(settings.db_path)
    await init_db(db)
    app.state.db = db
    app.state.llm = LLMClient(settings)
    yield
    await db.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM-Powered Test Failure Analyzer",
        version="0.1.0",
        description="Classifies pytest failures via Claude API with structured output and fallback heuristics.",
        lifespan=lifespan,
    )
    app.add_middleware(TimingMiddleware)
    app.include_router(analyze.router, prefix="/api/v1")
    app.include_router(evaluate.router, prefix="/api/v1")
    app.include_router(health.router)
    return app
