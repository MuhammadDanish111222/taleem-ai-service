from fastapi import APIRouter, Depends, Header
from typing import Optional
from app.core.config import Settings, get_settings
from app.core.security import verify_firebase_token
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health")
async def health_check(settings: Settings = Depends(get_settings)):
    return {"status": "ok", "service_name": settings.APP_NAME}

@router.get("/ready")
async def readiness_check(authorization: Optional[str] = Header(None)):
    uid = None
    if authorization:
        try:
            auth_context = verify_firebase_token(authorization)
            uid = auth_context.uid
            logger.info(f"Readiness check called with token for uid: {uid}")
        except Exception as e:
            logger.warning(f"Optional token verification failed on /ready: {e}")
            
    response = {"status": "ready"}
    if uid:
        response["uid"] = uid
    return response
