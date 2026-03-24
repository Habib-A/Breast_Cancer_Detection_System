# ── Build stage ───────────────────────────────────────────────────────────────
# Use CPU-only PyTorch to keep the image lean (~1.5GB vs ~5GB for CUDA)
FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids pulling CUDA wheels)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    python-multipart==0.0.9 \
    streamlit==1.35.0 \
    requests==2.32.2 \
    Pillow==10.3.0

# Copy application code
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY models/ ./models/

# Copy startup script
COPY start.sh .
RUN chmod +x start.sh

# Expose both ports
EXPOSE 8000 8501

# Railway injects $PORT — we use 8000 for FastAPI as the primary service.
# Streamlit runs alongside on 8501.
CMD ["./start.sh"]
