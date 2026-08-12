import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, status
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.core.internal_auth import verify_internal_jwt
from app.db.pool import get_db_connection

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
            auth_context = await verify_internal_jwt(authorization)
            uid = auth_context.uid
            logger.info(f"Readiness check called with token for uid: {uid}")
        except Exception as e:
            logger.warning(f"Optional token verification failed on /ready: {e}")

    try:
        async with get_db_connection() as connection:
            await connection.execute("SELECT 1")
    except Exception:
        logger.warning("Database readiness check failed.", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready"},
        )

    response = {"status": "ready"}
    if uid:
        response["uid"] = uid
    return response
