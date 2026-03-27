# CPU-only PyTorch image (~1.5GB vs ~5GB for CUDA)
FROM python:3.11-slim-bookworm

WORKDIR /app

# System deps for Pillow / torch
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

# Remaining dependencies. PyTorch may install NumPy 2 first; pip often leaves it in place unless we reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    python-multipart==0.0.9 \
    streamlit==1.35.0 \
    requests==2.32.2 \
    matplotlib==3.8.4 \
    Pillow==10.3.0 \
 && pip install --no-cache-dir --force-reinstall numpy==1.26.4

# Copy application code (repo uses capitalized folder names)
COPY App/ ./App/
COPY Frontend/ ./Frontend/
# Model weights are gitignored; mount a volume at /app/Model or set MODEL_PATH
RUN mkdir -p Model

COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000 8501

CMD ["./start.sh"]


# force rebuild
RUN echo "rebuild"
