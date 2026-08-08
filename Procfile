web: sh -c 'if [ "${TALEEM_PROCESS_ROLE:-api}" = "worker" ]; then exec uv run --no-dev python -m app.workers.main; else exec uv run --no-dev uvicorn app.main:app --host 0.0.0.0 --port "$PORT"; fi'
