from fastapi import FastAPI
from app.api.v1 import health

app = FastAPI(
    title="Taleem AI Service",
    description="Backend AI service for the Taleem AI platform"
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
