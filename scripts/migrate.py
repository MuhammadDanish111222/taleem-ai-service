import asyncio
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncpg  # noqa: E402

from app.db.migrator import run_migrations  # noqa: E402


async def main():
    load_dotenv()
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgrespassword@localhost:5432/taleem_dev",
    )
    conn = await asyncpg.connect(db_url, statement_cache_size=0)
    try:
        await run_migrations(conn)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
