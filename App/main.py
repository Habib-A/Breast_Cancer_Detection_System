from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from App.model import load_model, predict
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once when the server starts."""
    load_model()
    yield
    # Cleanup (if needed) goes here


app = FastAPI(
    title="Breast Histopathology API",
    description="ResNet50-based malignancy detection from histopathology patches",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this in production if needed
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Railway uses this to verify the service is alive."""
    return {"status": "ok", "service": "breast-histopathology-api"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Accept a histopathology image and return malignancy prediction.

    - **file**: PNG or JPEG histopathology image (ideally 224x224 patch)
    - Returns: prediction label, confidence %, and per-class probabilities
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=422,
            detail="Only JPEG and PNG images are accepted."
        )

    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        result = predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return {
        "filename": file.filename,
        **result
    }


if __name__ == "__main__":
    uvicorn.run("App.main:app", host="0.0.0.0", port=8000, reload=False)
