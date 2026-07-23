import asyncio
import os
import asyncpg
from app.db.migrator import run_migrations

async def main():
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/taleem_dev")
    conn = await asyncpg.connect(db_url)
    try:
        await run_migrations(conn)
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
