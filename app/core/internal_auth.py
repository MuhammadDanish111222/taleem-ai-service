import json
import logging
import time
import jwt
import redis
from fastapi import Header, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from app.core.config import get_settings

logger = logging.getLogger(__name__)

class AuthContext(BaseModel):
    uid: str
    is_admin: bool
    feature: str
    request_id: str

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
        return json.loads(keys_json)
    except Exception as e:
        logger.error(f"Failed to parse INTERNAL_JWT_PUBLIC_KEYS_JSON: {e}")
        return {}

def verify_internal_jwt(authorization: Optional[str] = Header(None)) -> AuthContext:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_INVALID_TOKEN", "message": "Missing or invalid authorization header"}
        )
    
    token = authorization.split("Bearer ")[1]
    
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        if not kid or not isinstance(kid, str):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_INVALID_TOKEN", "message": "Missing or invalid kid in token header"}
            )
            
        keys = get_public_keys()
        public_key = keys.get(kid)
        if not public_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_INVALID_TOKEN", "message": "Unknown kid"}
            )
            
        decoded_token = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience="taleem-ai-service",
            issuer="taleem-web"
        )
        
        # Mandatory claims and strict type checks
        uid = decoded_token.get("uid")
        if not uid or not isinstance(uid, str):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_INVALID_TOKEN", "message": "Missing or invalid uid claim"}
            )

        admin = decoded_token.get("admin")
        if admin is None or not isinstance(admin, bool):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_INVALID_TOKEN", "message": "Missing or invalid admin claim"}
            )

        feature = decoded_token.get("feature")
        if not feature or not isinstance(feature, str):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_INVALID_TOKEN", "message": "Missing or invalid feature claim"}
            )

        request_id = decoded_token.get("request_id")
        if not request_id or not isinstance(request_id, str):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_INVALID_TOKEN", "message": "Missing or invalid request_id claim"}
            )

        jti = decoded_token.get("jti")
        if not jti or not isinstance(jti, str):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_INVALID_TOKEN", "message": "Missing or invalid jti claim"}
            )

        iat = decoded_token.get("iat")
        exp = decoded_token.get("exp")

        if iat is None or not isinstance(iat, (int, float)):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_INVALID_TOKEN", "message": "Missing or invalid iat claim"}
            )

        if exp is None or not isinstance(exp, (int, float)):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_INVALID_TOKEN", "message": "Missing or invalid exp claim"}
            )

        if exp <= iat:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_INVALID_TOKEN", "message": "Token exp must be after iat"}
            )

        if (exp - iat) > 60:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_INVALID_TOKEN", "message": "Token TTL exceeds maximum 60s"}
            )
            
        # Atomic Redis replay prevention
        try:
            redis_client = get_redis()
            redis_key = f"jwt:jti:{jti}"
            # Atomic set-if-not-exists with 60s expiration
            is_new = redis_client.set(redis_key, "1", nx=True, ex=60)
            if not is_new:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={"code": "AUTH_REPLAY_DETECTED", "message": "Token replay detected"}
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Redis connection error during token verification: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_REDIS_ERROR", "message": "Redis failure during replay check"}
            )
        
        return AuthContext(
            uid=uid,
            is_admin=admin,
            feature=feature,
            request_id=request_id
        )
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_EXPIRED_TOKEN", "message": "Expired token"}
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_INVALID_TOKEN", "message": f"Invalid token: {e}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal auth error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_INVALID_TOKEN", "message": "Authentication failed"}
        )
