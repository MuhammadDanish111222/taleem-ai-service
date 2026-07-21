import json
import logging
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
    feature: Optional[str] = None
    request_id: Optional[str] = None

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

def verify_internal_jwt(authorization: str = Header(...)) -> AuthContext:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_INVALID_TOKEN", "message": "Missing or invalid authorization header"}
        )
    
    token = authorization.split("Bearer ")[1]
    
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        if not kid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_INVALID_TOKEN", "message": "Missing kid in token header"}
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
        
        jti = decoded_token.get("jti")
        if not jti:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_INVALID_TOKEN", "message": "Missing jti"}
            )
            
        redis_client = get_redis()
        redis_key = f"jwt:jti:{jti}"
        if redis_client.exists(redis_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "AUTH_REPLAY_DETECTED", "message": "Token replay detected"}
            )
            
        # Store JTI with 60 second TTL
        redis_client.setex(redis_key, 60, "1")
        
        return AuthContext(
            uid=decoded_token.get("uid"),
            is_admin=bool(decoded_token.get("admin", False)),
            feature=decoded_token.get("feature"),
            request_id=decoded_token.get("request_id")
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
