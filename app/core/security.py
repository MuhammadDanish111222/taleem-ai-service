import logging

from fastapi import Header, HTTPException, status
from firebase_admin import auth as firebase_auth
from pydantic import BaseModel

from app.core.firebase_admin import get_auth

logger = logging.getLogger(__name__)


class AuthContext(BaseModel):
    uid: str
    is_admin: bool


def verify_firebase_token(authorization: str = Header(...)) -> AuthContext:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "AUTH_INVALID_TOKEN",
                "message": "Missing or invalid authorization header",
            },
        )

    token = authorization.split("Bearer ")[1]

    try:
        auth_client = get_auth()
        decoded_token = auth_client.verify_id_token(token)
        uid = decoded_token.get("uid")
        is_admin = bool(decoded_token.get("admin", False))
        return AuthContext(uid=uid, is_admin=is_admin)
    except firebase_auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_INVALID_TOKEN", "message": "Invalid token"},
        )
    except firebase_auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_INVALID_TOKEN", "message": "Expired token"},
        )
    except Exception as e:
        logger.error(f"Firebase auth error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_INVALID_TOKEN", "message": "Authentication failed"},
        )
