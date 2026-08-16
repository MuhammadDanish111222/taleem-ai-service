from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.internal_auth import AuthContext, verify_internal_jwt
from app.db.pool import get_db_connection
from app.services.operations_dashboard import WINDOWS, OperationsDashboardService

router = APIRouter()


def admin(auth: AuthContext):
    if not auth.is_admin or auth.feature != "local_operations_dashboard":
        raise HTTPException(403, detail={"code": "FORBIDDEN_NOT_ADMIN"})


def _safe_target_id(value: str) -> str:
    """Audit target IDs can be opaque external identifiers; only UUIDs are display-safe."""
    try:
        return str(UUID(value))
    except (TypeError, ValueError, AttributeError):
        return "redacted"


@router.get("/internal/admin/operations-dashboard")
async def dashboard(
    window: str = Query("24h"), auth: AuthContext = Depends(verify_internal_jwt)
):
    admin(auth)
    if window not in WINDOWS:
        raise HTTPException(400, detail={"code": "OPERATIONS_WINDOW_INVALID"})
    async with get_db_connection() as conn:
        return await OperationsDashboardService(conn).dashboard(window)  # type: ignore[arg-type]


@router.get("/internal/admin/audit-search")
async def audit_search(
    window: str = Query("24h"),
    limit: int = Query(50, ge=1, le=100),
    cursor: UUID | None = None,
    action: str | None = Query(None, max_length=120),
    target_type: str | None = Query(None, max_length=120),
    target_id: str | None = Query(None, max_length=160),
    error_code: str | None = Query(None, max_length=120),
    auth: AuthContext = Depends(verify_internal_jwt),
):
    admin(auth)
    if window not in WINDOWS:
        raise HTTPException(400, detail={"code": "OPERATIONS_WINDOW_INVALID"})
    async with get_db_connection() as conn:
        rows = await conn.fetch(
            """SELECT id,action,target_type,target_id,created_at FROM admin_audit_logs
          WHERE created_at >= NOW() - $1::int * interval '1 second'
            AND ($2::text IS NULL OR action=$2) AND ($3::text IS NULL OR target_type=$3)
            AND ($4::text IS NULL OR target_id=$4)
            AND ($5::text IS NULL OR after_value->>'error_code'=$5)
            AND ($6::uuid IS NULL OR (created_at,id) < (SELECT created_at,id FROM admin_audit_logs WHERE id=$6::uuid))
          ORDER BY created_at DESC,id DESC LIMIT $7""",
            int(WINDOWS[window].total_seconds()),
            action,
            target_type,
            target_id,
            error_code,
            str(cursor) if cursor else None,
            limit,
        )
    values = [
        {
            "id": str(r["id"]),
            "action": r["action"],
            "target_type": r["target_type"],
            "target_id": _safe_target_id(r["target_id"]),
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]
    return {
        "items": values,
        "next_cursor": values[-1]["id"] if len(values) == limit else None,
    }
