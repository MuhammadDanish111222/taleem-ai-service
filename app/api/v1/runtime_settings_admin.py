"""Independent AI-service authorization for local runtime settings."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

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
async def list_runtime_settings(
    key: str | None = Query(default=None, max_length=160),
    scope_kind: str | None = Query(default=None),
    subject_id: str | None = Query(default=None, max_length=120),
    class_id: str | None = Query(default=None, max_length=120),
    account_tier: str | None = Query(default=None),
    auth: AuthContext = Depends(verify_internal_jwt),
):
    _admin(auth)
    async with get_db_connection() as conn:
        service = RuntimeSettingsService(conn)
        result = {
            "registry": service.metadata(),
            "effective": await service.list_effective(),
        }
        # A selected bounded scope is intentionally a point read, not scope enumeration.
        if key is not None or scope_kind is not None:
            if not key or not scope_kind:
                raise HTTPException(
                    status_code=400, detail={"code": "RUNTIME_SETTING_SCOPE_INVALID"}
                )
            try:
                scope = Scope(
                    kind=scope_kind,
                    subject_id=subject_id,
                    class_id=class_id,
                    account_tier=account_tier,
                )  # type: ignore[arg-type]
                definition = service._definition(key)
                service._validate(definition, definition.default, scope)
                result["selected"] = {
                    "key": key,
                    "scope": scope.as_dict(),
                    "value": await service.get(key, scope),
                }
            except RuntimeSettingError as exc:
                raise HTTPException(
                    status_code=400, detail={"code": str(exc)}
                ) from None
        return result


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
