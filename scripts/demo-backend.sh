#!/usr/bin/env bash
# Start only the local API. Never starts/restarts vLLM or GPU infrastructure.
set -euo pipefail
cd "$(dirname "$0")/.."
export HIVE_LIVE_AGENTS=true HIVE_GRAPH_VERSION=v2 HIVE_SPECIALIST_ROUTING=true
export HIVE_PLANNING_HORIZON=3 HIVE_PHYSICS_VERSION=v1
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8001/v1}"
export HIVE_MODEL=Qwen/Qwen2.5-7B-Instruct
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"
curl --fail --silent --show-error --max-time 5 "$OPENAI_API_BASE/models" >/dev/null
exec backend/.venv/bin/python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port "${HIVE_API_PORT:-8000}"
