import os
import subprocess
import json
import pytest
from unittest.mock import patch, MagicMock
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from app.core.internal_auth import verify_internal_jwt

# Generate a real RSA key pair for cross-repo signing integration test
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

WEB_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "taleem-web"))

@pytest.fixture
def mock_redis():
    with patch('app.core.internal_auth.get_redis') as mock_get_redis:
        mock_client = MagicMock()
        mock_client.set.return_value = True
        mock_get_redis.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_keys():
    with patch('app.core.internal_auth.get_public_keys') as mock_get_keys:
        mock_get_keys.return_value = {"integration-kid": public_pem}
        yield mock_get_keys

def test_ts_signer_to_python_verifier_integration(mock_keys, mock_redis):
    """End-to-end integration test: TypeScript signer (taleem-web) -> Python verifier (taleem-ai-service)."""
    if not os.path.exists(WEB_REPO_DIR):
        pytest.skip("taleem-web repository not found at " + WEB_REPO_DIR)
    env = os.environ.copy()
    env["INTERNAL_JWT_PRIVATE_KEY"] = private_pem
    env["INTERNAL_JWT_KEY_ID"] = "integration-kid"

    cmd = [
        "npx", "tsx", "scripts/sign_token_cli.ts",
        "ts-user-999", "true", "admin_ingestion", "req-cross-repo-123"
    ]

    result = subprocess.run(
        cmd,
        cwd=WEB_REPO_DIR,
        capture_output=True,
        text=True,
        env=env,
        shell=True
    )

    assert result.returncode == 0, f"TypeScript token generation failed: {result.stderr}"
    token = result.stdout.strip()
    assert token.count(".") == 2, f"Invalid JWT string output: {token}"

    # Verify token using Python verifier
    ctx = verify_internal_jwt(f"Bearer {token}")

    assert ctx.uid == "ts-user-999"
    assert ctx.is_admin is True
    assert ctx.feature == "admin_ingestion"
    assert ctx.request_id == "req-cross-repo-123"

    # Assert Redis recorded the exact JTI with 60s expiration using atomic SET NX EX
    mock_redis.set.assert_called_once()
    call_args = mock_redis.set.call_args
    assert call_args[0][0].startswith("jwt:jti:")
    assert call_args[1]["nx"] is True
    assert call_args[1]["ex"] == 60
