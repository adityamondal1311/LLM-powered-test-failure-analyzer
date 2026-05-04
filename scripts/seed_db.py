"""Initialize the SQLite database schema."""
from __future__ import annotations

import asyncio
import sys

import aiosqlite

from analyzer.pipeline.storage import init_db


async def main(db_path: str = "analyzer.db") -> None:
    async with aiosqlite.connect(db_path) as db:
        await init_db(db)
    print(f"Database initialized at: {db_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "analyzer.db"
    asyncio.run(main(path))
