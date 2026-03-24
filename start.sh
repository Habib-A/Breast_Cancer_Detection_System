#!/bin/bash
# Start FastAPI in the background
uvicorn App.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit — BACKEND_URL env var tells it where FastAPI lives
streamlit run Frontend/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false

# Keep container alive; if either process dies the container exits
wait
