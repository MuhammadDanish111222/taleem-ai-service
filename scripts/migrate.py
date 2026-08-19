import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio

import asyncpg

from dotenv import load_dotenv

load_dotenv()

from app.db.migrator import run_migrations


async def main():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not found in environment or .env")
        sys.exit(1)
    conn = await asyncpg.connect(db_url, statement_cache_size=0)
    try:
        await run_migrations(conn)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
