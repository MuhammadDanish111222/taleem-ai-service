import os
import sys

# Ensure app package is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import patch, MagicMock

if len(sys.argv) < 3:
    print("Usage: python run_fastapi_server.py <key_id> <port>", file=sys.stderr)
    sys.exit(1)

key_id = sys.argv[1]
port = int(sys.argv[2])
public_pem = os.environ.get("MOCK_PUBLIC_KEY_PEM") or sys.stdin.read()

mock_redis = MagicMock()
mock_redis.set.return_value = True

patch('app.core.internal_auth.get_redis', return_value=mock_redis).start()
patch('app.core.internal_auth.get_public_keys', return_value={key_id: public_pem}).start()

from app.main import app
import uvicorn

uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")
