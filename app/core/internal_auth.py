import asyncio
import hashlib
import hmac
import json
import logging
from typing import Optional

import jwt
import redis
from fastapi import Header, HTTPException, status
from pydantic import BaseModel

from app.core.config import get_settings
from app.db.pool import get_db_connection
from app.repositories.audit_repository import AuditRepository

logger = logging.getLogger(__name__)


class AuthContext(BaseModel):
    uid: str
    is_admin: bool
    feature: str
    request_id: str
    account_tier: Optional[str] = None


_redis_client = None


def get_redis():
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


def get_public_keys():
    settings = get_settings()
    try:
        keys_json = settings.INTERNAL_JWT_PUBLIC_KEYS_JSON
        if not keys_json or keys_json.strip() == "":
            keys_json = "{}"
        parsed_keys = json.loads(keys_json)
        if not isinstance(parsed_keys, dict):
            raise ValueError("INTERNAL_JWT_PUBLIC_KEYS_JSON must contain an object")

        # Environment files commonly carry PEM line breaks as literal ``\\n``
        # sequences.  PyJWT needs real line breaks before it can parse the PEM.
        return {
            key_id: public_key.replace("\\n", "\n")
            for key_id, public_key in parsed_keys.items()
            if isinstance(key_id, str) and isinstance(public_key, str)
        }
    except Exception as e:
        logger.error(f"Failed to parse INTERNAL_JWT_PUBLIC_KEYS_JSON: {e}")
        return {}


def _jti_hash(jti: str) -> str:
    secret = get_settings().INTERNAL_JTI_HMAC_SECRET
    if not secret:
        raise RuntimeError("INTERNAL_JTI_HASH_SECRET_UNAVAILABLE")
    return hmac.new(
        secret.encode("utf-8"), jti.encode("utf-8"), hashlib.sha256
    ).hexdigest()


async def _record_jti_postgres(
    *, jti_hash: str, expires_at: float, fallback_event: bool
) -> bool:
    """Atomically mirror/claim a JTI; False means it was already consumed."""
    async with get_db_connection() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM internal_jti_replay WHERE expires_at <= NOW()"
            )
            inserted = await conn.fetchval(
                """INSERT INTO internal_jti_replay(jti_hash, expires_at)
                   VALUES($1, to_timestamp($2))
                   ON CONFLICT(jti_hash) DO NOTHING
                   RETURNING TRUE""",
                jti_hash,
                expires_at,
            )
            if fallback_event:
                await AuditRepository(conn).create_audit_log(
                    actor_id="system",
                    action="auth.redis_fallback",
                    target_type="internal_jti",
                    target_id=jti_hash,
                    after_value={"event": "redis_unavailable"},
                )
            return bool(inserted)


async def verify_internal_jwt(
    authorization: Optional[str] = Header(None),
) -> AuthContext:
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
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        if not kid or not isinstance(kid, str):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "AUTH_INVALID_TOKEN",
                    "message": "Missing or invalid kid in token header",
                },
            )

        keys = get_public_keys()
        public_key = keys.get(kid)
        if not public_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_INVALID_TOKEN", "message": "Unknown kid"},
            )

        decoded_token = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience="taleem-ai-service",
            issuer="taleem-web",
        )

        # Mandatory claims and strict type checks
        uid = decoded_token.get("uid")
        if not uid or not isinstance(uid, str):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "AUTH_INVALID_TOKEN",
                    "message": "Missing or invalid uid claim",
                },
            )

        admin = decoded_token.get("admin")
        if admin is None or not isinstance(admin, bool):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "AUTH_INVALID_TOKEN",
                    "message": "Missing or invalid admin claim",
                },
            )

        feature = decoded_token.get("feature")
        if not feature or not isinstance(feature, str):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "AUTH_INVALID_TOKEN",
                    "message": "Missing or invalid feature claim",
                },
            )

        request_id = decoded_token.get("request_id")
        if not request_id or not isinstance(request_id, str):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "AUTH_INVALID_TOKEN",
                    "message": "Missing or invalid request_id claim",
                },
            )

        jti = decoded_token.get("jti")
        if not jti or not isinstance(jti, str):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "AUTH_INVALID_TOKEN",
                    "message": "Missing or invalid jti claim",
                },
            )

        iat = decoded_token.get("iat")
        exp = decoded_token.get("exp")

        if iat is None or not isinstance(iat, (int, float)):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "AUTH_INVALID_TOKEN",
                    "message": "Missing or invalid iat claim",
                },
            )

        if exp is None or not isinstance(exp, (int, float)):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "AUTH_INVALID_TOKEN",
                    "message": "Missing or invalid exp claim",
                },
            )

        if exp <= iat:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "AUTH_INVALID_TOKEN",
                    "message": "Token exp must be after iat",
                },
            )

        if (exp - iat) > 60:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "AUTH_INVALID_TOKEN",
                    "message": "Token TTL exceeds maximum 60s",
                },
            )

        account_tier = decoded_token.get("account_tier")
        if account_tier is not None and account_tier not in {
            "anonymous",
            "google",
            "premium",
        }:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "AUTH_INVALID_TOKEN",
                    "message": "Invalid account_tier claim",
                },
            )
        if feature in {"ask", "ask_usage", "multiple_ask"} and account_tier is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "AUTH_INVALID_TOKEN",
                    "message": "Missing account_tier claim",
                },
            )

        # Redis is the normal replay path. PostgreSQL mirrors every accepted JTI,
        # so an outage transition cannot reopen a token already consumed in Redis.
        redis_available = True
        try:
            redis_client = get_redis()
            redis_key = f"jwt:jti:{jti}"
            is_new = await asyncio.to_thread(
                redis_client.set, redis_key, "1", nx=True, ex=60
            )
            if not is_new:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={
                        "code": "AUTH_REPLAY_DETECTED",
                        "message": "Token replay detected",
                    },
                )
        except HTTPException:
            raise
        except Exception:
            redis_available = False
            logger.warning(
                "internal_jti_redis_fallback",
                extra={"event": "redis_unavailable"},
            )

        try:
            pg_is_new = await _record_jti_postgres(
                jti_hash=_jti_hash(jti),
                expires_at=float(exp),
                fallback_event=not redis_available,
            )
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "AUTH_REPLAY_PROTECTION_UNAVAILABLE",
                    "message": "Replay protection is unavailable",
                },
            )
        if not pg_is_new:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "AUTH_REPLAY_DETECTED",
                    "message": "Token replay detected",
                },
            )

        return AuthContext(
            uid=uid,
            is_admin=admin,
            feature=feature,
            request_id=request_id,
            account_tier=account_tier,
        )

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_EXPIRED_TOKEN", "message": "Expired token"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_INVALID_TOKEN", "message": "Invalid token"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "internal_auth_error",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_INVALID_TOKEN", "message": "Authentication failed"},
        )
