#!/usr/bin/env bash
set -euo pipefail

# FastAPI on fixed port inside the container; Streamlit binds Railway's $PORT for public traffic.
API_PORT="${API_PORT:-8000}"
STREAMLIT_PORT="${PORT:-8501}"

mkdir -p Model

# Weights are not in git. For Railway: set MODEL_DOWNLOAD_URL to a direct HTTPS link to best_model.pth,
# or attach a volume with the file at /app/Model/best_model.pth.
if [ -n "${MODEL_DOWNLOAD_URL:-}" ] && [ ! -f Model/best_model.pth ]; then
  echo "Downloading model weights..."
  curl -fL --connect-timeout 30 --max-time 900 "${MODEL_DOWNLOAD_URL}" -o Model/best_model.pth
fi

export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:${API_PORT}}"

uvicorn App.main:app --host 0.0.0.0 --port "${API_PORT}" &
UVICORN_PID=$!

echo "Waiting for FastAPI on port ${API_PORT}..."
READY=0
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${API_PORT}/health" > /dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "${UVICORN_PID}" 2>/dev/null; then
    echo "ERROR: uvicorn exited before the API became healthy."
    echo "Common causes: missing Model/best_model.pth (set MODEL_DOWNLOAD_URL or mount a volume), or out of memory."
    exit 1
  fi
  sleep 1
done

if [ "${READY}" -ne 1 ]; then
  echo "ERROR: Timed out waiting for /health. Is the model file present and valid?"
  exit 1
fi

echo "FastAPI is ready. Starting Streamlit on port ${STREAMLIT_PORT}..."

streamlit run Frontend/streamlit_app.py \
    --server.port "${STREAMLIT_PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false

wait "${UVICORN_PID}"
