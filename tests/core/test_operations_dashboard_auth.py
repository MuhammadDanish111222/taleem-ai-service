import pytest
from fastapi import HTTPException

from app.api.v1.operations_dashboard import admin
from app.core.internal_auth import AuthContext
from app.services.operations_dashboard import OperationsDashboardService


@pytest.mark.parametrize(
    "is_admin,feature",
    [(False, "local_operations_dashboard"), (True, "local_runtime_settings")],
)
def test_operations_dashboard_requires_exact_admin_feature(is_admin, feature):
    with pytest.raises(HTTPException) as exc_info:
        admin(
            AuthContext(
                uid="admin", is_admin=is_admin, feature=feature, request_id="request"
            )
        )
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == {"code": "FORBIDDEN_NOT_ADMIN"}


@pytest.mark.asyncio
async def test_operations_dashboard_rejects_unbounded_window():
    with pytest.raises(ValueError, match="OPERATIONS_WINDOW_INVALID"):
        await OperationsDashboardService(None).dashboard("forever")  # type: ignore[arg-type]
