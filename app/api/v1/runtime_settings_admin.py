"""Independent AI-service authorization for local runtime settings."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.internal_auth import AuthContext, verify_internal_jwt
from app.db.pool import get_db_connection
from app.schemas.runtime_settings import RuntimeSettingMutation
from app.services.runtime_settings import (
    RuntimeSettingError,
    RuntimeSettingsService,
    Scope,
)

router = APIRouter()


def _admin(auth: AuthContext) -> None:
    if not auth.is_admin or auth.feature != "local_runtime_settings":
        raise HTTPException(status_code=403, detail={"code": "FORBIDDEN_NOT_ADMIN"})


@router.get("/internal/admin/runtime-settings")
async def list_runtime_settings(auth: AuthContext = Depends(verify_internal_jwt)):
    _admin(auth)
    async with get_db_connection() as conn:
        service = RuntimeSettingsService(conn)
        return {
            "registry": service.metadata(),
            "effective": await service.list_effective(),
        }


@router.post("/internal/admin/runtime-settings")
async def set_runtime_setting(
    request: RuntimeSettingMutation, auth: AuthContext = Depends(verify_internal_jwt)
):
    _admin(auth)
    try:
        async with get_db_connection() as conn:
            return await RuntimeSettingsService(conn).set(
                key=request.key,
                scope=Scope.from_dict(request.scope.model_dump()),
                value=request.value,
                actor_id=auth.uid,
                request_id=auth.request_id,
            )
    except RuntimeSettingError as exc:
        raise HTTPException(status_code=400, detail={"code": str(exc)}) from None
