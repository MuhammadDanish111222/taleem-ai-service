import pytest
import jwt
import time
from fastapi import HTTPException
from app.core.internal_auth import verify_internal_jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from unittest.mock import patch, MagicMock

# Generate a temporary RSA key pair for testing
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
public_key = private_key.public_key()

private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
).decode('utf-8')

public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
).decode('utf-8')

def create_token(kid="test-key", uid="user123", exp_delta=60, aud="taleem-ai-service", jti="jti-123"):
    payload = {
        "uid": uid,
        "admin": False,
        "feature": "test",
        "request_id": "req-123",
        "aud": aud,
        "iss": "taleem-web",
        "jti": jti,
        "iat": int(time.time()),
        "exp": int(time.time()) + exp_delta
    }
    return jwt.encode(payload, private_pem, algorithm="RS256", headers={"kid": kid})

@pytest.fixture
def mock_redis():
    with patch('app.core.internal_auth.get_redis') as mock_get_redis:
        mock_client = MagicMock()
        mock_client.exists.return_value = False
        mock_get_redis.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_keys():
    with patch('app.core.internal_auth.get_public_keys') as mock_get_keys:
        mock_get_keys.return_value = {"test-key": public_pem}
        yield mock_get_keys

def test_missing_token():
    with pytest.raises(HTTPException) as exc_info:
        verify_internal_jwt("")
    assert exc_info.value.status_code == 401
    assert "Missing or invalid authorization header" in str(exc_info.value.detail)

def test_expired_token(mock_keys, mock_redis):
    token = create_token(exp_delta=-10)
    with pytest.raises(HTTPException) as exc_info:
        verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["code"] == "AUTH_EXPIRED_TOKEN"

def test_wrong_audience(mock_keys, mock_redis):
    token = create_token(aud="wrong-audience")
    with pytest.raises(HTTPException) as exc_info:
        verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401
    assert "Invalid token" in str(exc_info.value.detail)

def test_wrong_signature(mock_keys, mock_redis):
    # Create token with a different private key
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other_pem = other_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ).decode('utf-8')
    payload = {"uid": "user123", "aud": "taleem-ai-service", "iss": "taleem-web", "exp": int(time.time())+60}
    token = jwt.encode(payload, other_pem, algorithm="RS256", headers={"kid": "test-key"})
    
    with pytest.raises(HTTPException) as exc_info:
        verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401
    assert "Invalid token" in str(exc_info.value.detail)

def test_missing_claims(mock_keys, mock_redis):
    # Missing jti
    payload = {"uid": "user123", "aud": "taleem-ai-service", "iss": "taleem-web", "exp": int(time.time())+60}
    token = jwt.encode(payload, private_pem, algorithm="RS256", headers={"kid": "test-key"})
    
    with pytest.raises(HTTPException) as exc_info:
        verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401
    assert "Missing jti" in str(exc_info.value.detail)

def test_replayed_jti(mock_keys, mock_redis):
    mock_redis.exists.return_value = True
    token = create_token(jti="used-jti")
    with pytest.raises(HTTPException) as exc_info:
        verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["code"] == "AUTH_REPLAY_DETECTED"

def test_valid_token(mock_keys, mock_redis):
    token = create_token()
    ctx = verify_internal_jwt(f"Bearer {token}")
    assert ctx.uid == "user123"
    assert ctx.is_admin == False
    mock_redis.exists.assert_called_once_with("jwt:jti:jti-123")
    mock_redis.setex.assert_called_once_with("jwt:jti:jti-123", 60, "1")
