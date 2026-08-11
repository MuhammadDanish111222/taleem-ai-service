import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncpg

from app.core.config import get_settings


async def main():
    conn = await asyncpg.connect(get_settings().DATABASE_URL, statement_cache_size=0)
    try:
        print("=== MULTIPLE ASK JOBS ===")
        jobs = await conn.fetch(
            "SELECT * FROM multiple_ask_jobs ORDER BY created_at DESC LIMIT 1;"
        )
        for j in jobs:
            print(dict(j))

        print("\n=== MULTIPLE ASK JOB ITEMS ===")
        items = await conn.fetch(
            "SELECT * FROM multiple_ask_job_items ORDER BY created_at DESC LIMIT 12;"
        )
        for it in items:
            print(dict(it))
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
