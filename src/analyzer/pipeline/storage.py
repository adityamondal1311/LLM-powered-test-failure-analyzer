from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator

import aiosqlite

from analyzer.models.pipeline import ScoredResult, StoredRecord

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS analysis_results (
    record_id       TEXT PRIMARY KEY,
    test_id         TEXT NOT NULL,
    category        TEXT NOT NULL,
    confidence      REAL NOT NULL,
    rank_score      REAL NOT NULL,
    fallback_source TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    stored_at_utc   REAL NOT NULL
);
"""

_CREATE_INDEX_CATEGORY = """
CREATE INDEX IF NOT EXISTS idx_category ON analysis_results(category);
"""

_CREATE_INDEX_TIME = """
CREATE INDEX IF NOT EXISTS idx_stored_at ON analysis_results(stored_at_utc DESC);
"""


async def init_db(db: aiosqlite.Connection) -> None:
    await db.execute(_CREATE_TABLE)
    await db.execute(_CREATE_INDEX_CATEGORY)
    await db.execute(_CREATE_INDEX_TIME)
    await db.commit()


async def store_result(
    result: ScoredResult, db: aiosqlite.Connection
) -> StoredRecord:
    record_id = str(uuid.uuid4())
    now = time.time()
    h = result.validation.inference.hypothesis
    test_id = result.validation.inference.parsed_failure.test_id

    await db.execute(
        """
        INSERT INTO analysis_results
            (record_id, test_id, category, confidence, rank_score,
             fallback_source, payload_json, stored_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record_id,
            test_id,
            h.category.value,
            result.final_confidence,
            result.rank_score,
            result.routed_to.value,
            result.model_dump_json(),
            now,
        ),
    )
    await db.commit()

    return StoredRecord(
        record_id=record_id,
        scored_result=result,
        stored_at_utc=now,
        db_path=str(db._connector._filename),  # type: ignore[attr-defined]
    )


async def store_batch(
    results: list[ScoredResult], db: aiosqlite.Connection
) -> list[StoredRecord]:
    records = []
    for r in results:
        records.append(await store_result(r, db))
    return records


async def query_by_category(
    category: str,
    db: aiosqlite.Connection,
    limit: int = 50,
) -> list[dict]:  # type: ignore[type-arg]
    async with db.execute(
        "SELECT payload_json FROM analysis_results WHERE category = ? "
        "ORDER BY stored_at_utc DESC LIMIT ?",
        (category, limit),
    ) as cursor:
        rows = await cursor.fetchall()
    return [json.loads(r[0]) for r in rows]


async def get_aggregate_stats(db: aiosqlite.Connection) -> dict:  # type: ignore[type-arg]
    async with db.execute(
        """
        SELECT
            COUNT(*) as total,
            AVG(confidence) as avg_conf,
            SUM(CASE WHEN fallback_source = 'heuristic' THEN 1 ELSE 0 END) as fallback_count
        FROM analysis_results
        """
    ) as cursor:
        row = await cursor.fetchone()

    total = row[0] or 0
    avg_conf = row[1] or 0.0
    fallback_count = row[2] or 0

    async with db.execute(
        "SELECT category, COUNT(*) FROM analysis_results GROUP BY category"
    ) as cursor:
        cat_rows = await cursor.fetchall()

    return {
        "total": total,
        "avg_confidence": round(avg_conf, 4),
        "fallback_rate": round(fallback_count / total, 4) if total else 0.0,
        "category_distribution": {r[0]: r[1] for r in cat_rows},
    }


async def open_db(db_path: str) -> AsyncGenerator[aiosqlite.Connection, None]:
    async with aiosqlite.connect(db_path) as db:
        await init_db(db)
        yield db
