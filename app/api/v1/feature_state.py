import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.internal_auth import AuthContext, verify_internal_jwt
from app.db.pool import get_db_connection
from app.services.runtime_settings import RuntimeSettingsService, Scope

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/internal/feature-state/{feature}")
async def get_feature_state(
    feature: str,
    auth: AuthContext = Depends(verify_internal_jwt),
):
    """Protected endpoint for reading feature lifecycle state for server-side page rendering."""
    if auth.feature != "feature_state_read":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "AUTH_FEATURE_FORBIDDEN",
                "message": "Feature read claim required",
            },
        )

    if feature != "multiple_ask":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "UNKNOWN_FEATURE",
                "message": f"Feature '{feature}' is not managed via this endpoint",
            },
        )

    try:
        async with get_db_connection() as conn:
            state = await RuntimeSettingsService(conn).get(
                "feature.multiple_ask", Scope(kind="global")
            )
            safe_state = (
                state if state in ("enabled", "coming_soon", "disabled") else "disabled"
            )
            return {"feature": feature, "state": safe_state}
    except Exception:
        logger.exception("Failed to query runtime feature state for %s", feature)
        return {"feature": feature, "state": "disabled"}
