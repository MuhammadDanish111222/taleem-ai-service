"""Signed internal endpoints for the public same-origin Ask BFF."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.core.internal_auth import AuthContext, verify_internal_jwt
from app.db.pool import get_db_connection
from app.repositories.ask_repository import AskRepository
from app.schemas.ask import AskRequest, AskResponse, UsageDto
from app.services.answers.generate import AskService, AskServiceError
from app.services.usage.models import AccountTier
from app.services.usage.service import UsageLimitExceeded, UsageService

router = APIRouter()


def _tier(auth: AuthContext) -> AccountTier:
    if auth.feature not in {"ask", "ask_usage"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "AUTH_FEATURE_FORBIDDEN", "message": "Feature denied"},
        )
    try:
        return AccountTier(auth.account_tier)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_INVALID_TOKEN", "message": "Invalid account tier"},
        ) from None


@router.post("/internal/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    auth: AuthContext = Depends(verify_internal_jwt),
):
    if str(request.request_id) != auth.request_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "REQUEST_ID_MISMATCH",
                "message": "Signed and payload request IDs differ",
            },
        )
    try:
        async with get_db_connection() as conn:
            return await AskService(conn).ask(request, uid=auth.uid, tier=_tier(auth))
    except UsageLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "USAGE_LIMIT_REACHED",
                "message": "Daily question limit reached",
                "usage": {
                    "feature": "single_question",
                    "used": exc.used,
                    "limit": exc.limit if exc.student_visible else None,
                    "remaining": 0 if exc.student_visible else None,
                    "resets_at": exc.resets_at.isoformat(),
                },
            },
        ) from None
    except AskServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": "Ask request could not be completed"},
        ) from None


@router.get("/internal/ask/usage", response_model=UsageDto)
async def ask_usage(auth: AuthContext = Depends(verify_internal_jwt)):
    try:
        async with get_db_connection() as conn:
            return await AskService(conn).usage(uid=auth.uid, tier=_tier(auth))
    except AskServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": "Usage is unavailable"},
        ) from None


@router.get("/internal/ask/visual/{visual_id}")
async def ask_visual_reference(
    visual_id: str = Path(min_length=1, max_length=160),
    auth: AuthContext = Depends(verify_internal_jwt),
):
    """Return a server-only storage reference for this Ask's reviewed visual."""

    if auth.feature != "ask_visual":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "AUTH_FEATURE_FORBIDDEN", "message": "Feature denied"},
        )
    try:
        uid_hash = UsageService.uid_hash(auth.uid)
        async with get_db_connection() as conn:
            reference = await AskRepository(conn).visual_stream_reference(
                client_request_id=auth.request_id,
                uid_hash=uid_hash,
                visual_id=visual_id,
            )
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "VISUAL_CONFIGURATION_ERROR",
                "message": "Visual service is unavailable",
            },
        ) from None
    if reference is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "VISUAL_NOT_FOUND", "message": "Visual not found"},
        )
    return reference
