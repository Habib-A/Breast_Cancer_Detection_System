#!/usr/bin/env bash
set -euo pipefail

# FastAPI listens on a fixed port inside the container; Streamlit uses Railway's $PORT when set.
API_PORT="${API_PORT:-8000}"
STREAMLIT_PORT="${PORT:-8501}"

export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:${API_PORT}}"

uvicorn App.main:app --host 0.0.0.0 --port "${API_PORT}" &

streamlit run Frontend/streamlit_app.py \
    --server.port "${STREAMLIT_PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false

wait
