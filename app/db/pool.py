"""Asyncpg Connection Pool Lifecycle & Transaction Helpers."""

import asyncpg
from pgvector.asyncpg import register_vector
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None

async def init_db_pool(dsn: Optional[str] = None) -> asyncpg.Pool:
    """Initializes the global asyncpg connection pool."""
    global _pool
    if _pool is not None:
        return _pool
        
    connection_url = dsn or get_settings().DATABASE_URL
    
    async def init_connection(conn: asyncpg.Connection):
        # Register pgvector type codec on every connection
        await register_vector(conn)
        
    _pool = await asyncpg.create_pool(
        dsn=connection_url,
        min_size=2,
        max_size=10,
        init=init_connection
    )
    logger.info("Database connection pool initialized successfully.")
    return _pool

async def close_db_pool() -> None:
    """Closes the global asyncpg connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed.")

def get_db_pool() -> asyncpg.Pool:
    """Retrieves the active global connection pool."""
    if _pool is None:
        raise RuntimeError("Database connection pool is not initialized.")
    return _pool

@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """Acquires a single connection from the pool."""
    pool = get_db_pool()
    async with pool.acquire() as conn:
        yield conn

@asynccontextmanager
async def transaction() -> AsyncGenerator[asyncpg.Connection, None]:
    """Acquires a connection and executes within an atomic database transaction."""
    pool = get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            yield conn
