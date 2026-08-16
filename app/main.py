import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1 import ask, ask_admin, health, internal, runtime_settings_admin
from app.db.pool import close_db_pool, init_db_pool

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize DB pool (non-blocking if DB isn't reached immediately)
    try:
        await init_db_pool()
    except Exception:
        logger.exception("Database pool initialization failed; service is not ready.")
    yield
    # Shutdown: close DB pool
    await close_db_pool()


app = FastAPI(
    title="Taleem AI Service",
    description="Backend AI service for the Taleem AI platform",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(internal.router, prefix="/api/v1", tags=["internal"])
app.include_router(ask.router, prefix="/api/v1", tags=["ask"])
app.include_router(ask_admin.router, prefix="/api/v1", tags=["ask-admin"])
app.include_router(
    runtime_settings_admin.router, prefix="/api/v1", tags=["runtime-settings-admin"]
)
