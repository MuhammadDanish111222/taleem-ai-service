from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.v1 import health, internal
from app.db.pool import init_db_pool, close_db_pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize DB pool (non-blocking if DB isn't reached immediately)
    try:
        await init_db_pool()
    except Exception:
        pass
    yield
    # Shutdown: close DB pool
    await close_db_pool()

app = FastAPI(
    title="Taleem AI Service",
    description="Backend AI service for the Taleem AI platform",
    lifespan=lifespan
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(internal.router, prefix="/api/v1", tags=["internal"])
