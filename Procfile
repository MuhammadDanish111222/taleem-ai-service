web: sh -c 'if [ "${TALEEM_PROCESS_ROLE:-api}" = "worker" ]; then exec uv run python -m app.workers.main; else exec uv run uvicorn app.main:app --host 0.0.0.0 --port "$PORT"; fi'
