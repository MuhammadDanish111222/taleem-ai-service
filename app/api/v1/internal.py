from fastapi import APIRouter, Depends
from app.core.internal_auth import verify_internal_jwt, AuthContext

router = APIRouter()

@router.get("/internal/verify")
async def verify_internal_access(auth_context: AuthContext = Depends(verify_internal_jwt)):
    """Protected internal endpoint used for verifying cross-repository identity propagation."""
    return {
        "status": "authenticated",
        "uid": auth_context.uid,
        "is_admin": auth_context.is_admin,
        "feature": auth_context.feature,
        "request_id": auth_context.request_id
    }
