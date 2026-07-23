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

os.environ.setdefault("POSTGRES_SERVER", "localhost")
os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "postgres")
os.environ.setdefault("POSTGRES_DB", "taleem_ai")
os.environ.setdefault("REDIS_HOST", "localhost")

import socket
import asyncio

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("0.0.0.0", port))
sock.listen(128)
sock.setblocking(False)
actual_port = sock.getsockname()[1]
print(f"SERVER_STARTED_PORT:{actual_port}", flush=True)

from app.main import app
import uvicorn

config = uvicorn.Config(app, host="0.0.0.0", port=actual_port, log_level="error")
server = uvicorn.Server(config)
asyncio.run(server.serve(sockets=[sock]))
