from fastapi import APIRouter, Depends
from app.core.config import Settings, get_settings

router = APIRouter()

@router.get("/health")
async def health_check(settings: Settings = Depends(get_settings)):
    return {"status": "ok", "service_name": settings.APP_NAME}

@router.get("/ready")
async def readiness_check():
    return {"status": "ready"}
