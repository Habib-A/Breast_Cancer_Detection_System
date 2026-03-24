#!/usr/bin/env bash
set -euo pipefail

# FastAPI on fixed port inside the container; Streamlit binds Railway's $PORT for public traffic.
API_PORT="${API_PORT:-8000}"
STREAMLIT_PORT="${PORT:-8501}"

mkdir -p Model

# Weights are not in git. On Railway set MODEL_DOWNLOAD_URL to a direct HTTPS link (e.g. GitHub Release asset).
# Optional: MODEL_DOWNLOAD_BEARER_TOKEN for private files (sent as Authorization: Bearer ...).
if [ -n "${MODEL_DOWNLOAD_URL:-}" ] && [ ! -f Model/best_model.pth ]; then
  echo "Downloading model weights..."
  CURL_HEADERS=()
  if [ -n "${MODEL_DOWNLOAD_BEARER_TOKEN:-}" ]; then
    CURL_HEADERS+=(-H "Authorization: Bearer ${MODEL_DOWNLOAD_BEARER_TOKEN}")
  fi
  curl -fL --connect-timeout 30 --max-time 900 \
    --retry 3 --retry-delay 5 --retry-all-errors \
    "${CURL_HEADERS[@]}" \
    -o Model/best_model.pth \
    "${MODEL_DOWNLOAD_URL}"
  if [ ! -s Model/best_model.pth ]; then
    echo "ERROR: Model download produced an empty file. Check MODEL_DOWNLOAD_URL (must be a direct file URL, not an HTML page)."
    exit 1
  fi
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
