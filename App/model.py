import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os

# Class labels matching your training setup
CLASS_NAMES = {0: "benign", 1: "malignant"}

# ImageNet normalization (same as your val_test_transform)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

_model = None
_device = None


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_model_loaded() -> bool:
    """Used by the backend /health endpoint to avoid serving requests early."""
    return _model is not None


def load_model() -> nn.Module:
    """Load ResNet50 with saved weights. Called once at startup."""
    global _model, _device

    _device = get_device()

    model_path = os.environ.get("MODEL_PATH", "Model/best_model.pth")
    if not os.path.exists(model_path):
        alt_model_path = "models/best_model.pth"
        if os.path.exists(alt_model_path):
            model_path = alt_model_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. "
            "Add best_model.pth (e.g. Railway volume mounted at /app/Model) or set MODEL_PATH."
        )

    # Re-create architecture (must match training setup)
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # Load saved weights
    state_dict = torch.load(model_path, map_location=_device)
    model.load_state_dict(state_dict)
    model.to(_device)
    model.eval()

    _model = model
    print(f"Model loaded from {model_path} on {_device}")
    return model


def predict(image_bytes: bytes) -> dict:
    """
    Run inference on raw image bytes.
    Returns label, confidence scores for both classes.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(_device)  # [1, 3, 224, 224]

    with torch.no_grad():
        outputs = _model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze()  # [2]

    benign_prob = probs[0].item()
    malignant_prob = probs[1].item()
    predicted_class = int(torch.argmax(probs).item())

    return {
        "prediction": CLASS_NAMES[predicted_class],
        "confidence": round(max(benign_prob, malignant_prob) * 100, 2),
        "probabilities": {
            "benign": round(benign_prob * 100, 2),
            "malignant": round(malignant_prob * 100, 2),
        },
    }
