import streamlit as st
import requests
import os
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Histopathology Classifier",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean medical aesthetic */
    .main { background-color: #f8f9fa; }
    .stApp { font-family: 'Source Sans Pro', sans-serif; }

    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .result-malignant {
        background: #fff5f5;
        border-color: #e53e3e;
        color: #742a2a;
    }
    .result-benign {
        background: #f0fff4;
        border-color: #38a169;
        color: #1c4532;
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .confidence-text {
        font-size: 1rem;
        opacity: 0.85;
    }
    .disclaimer {
        background: #fffbeb;
        border: 1px solid #f6e05e;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.82rem;
        color: #744210;
        margin-top: 1rem;
    }
    .prob-bar-label {
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── API config ────────────────────────────────────────────────────────────────
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{BACKEND_URL}/predict"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 Breast Histopathology Classifier")
st.markdown(
    "Upload a histopathology patch image to classify it as **benign** or **malignant** "
    "using a fine-tuned ResNet50 model trained on the BreakHis dataset."
)
st.divider()

# ── Image upload ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a histopathology image (PNG or JPEG)",
    type=["png", "jpg", "jpeg"],
    help="Ideally a 224×224 patch at 40×, 100×, 200×, or 400× magnification",
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("**Uploaded image**")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        st.caption(
            f"{uploaded_file.name} · {image.size[0]}×{image.size[1]}px · "
            f"{len(uploaded_file.getvalue()) / 1024:.1f} KB"
        )

    with col2:
        st.markdown("**Prediction**")

        with st.spinner("Running inference..."):
            try:
                response = requests.post(
                    PREDICT_ENDPOINT,
                    files={"file": (uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    uploaded_file.type)},
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot reach the backend API. "
                    f"Make sure FastAPI is running at `{BACKEND_URL}`."
                )
                st.stop()
            except requests.exceptions.Timeout:
                st.error("Request timed out. The model may be loading - try again in a moment.")
                st.stop()
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.status_code} - {e.response.text}")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.stop()

        prediction = result["prediction"]
        confidence = result["confidence"]
        probs = result["probabilities"]

        card_class = "result-malignant" if prediction == "malignant" else "result-benign"
        icon = "⚠️" if prediction == "malignant" else "✅"

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-title">{icon} {prediction.capitalize()}</div>
                <div class="confidence-text">Confidence: <strong>{confidence}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Class probabilities**")

        st.markdown('<p class="prob-bar-label">Benign</p>', unsafe_allow_html=True)
        st.progress(probs["benign"] / 100)
        st.caption(f"{probs['benign']}%")

        st.markdown('<p class="prob-bar-label">Malignant</p>', unsafe_allow_html=True)
        st.progress(probs["malignant"] / 100)
        st.caption(f"{probs['malignant']}%")

        st.markdown(
            '<div class="disclaimer">⚕️ <strong>For research purposes only.</strong> '
            'This tool is not a medical device and must not be used for clinical diagnosis. '
            'Always consult a qualified pathologist.</div>',
            unsafe_allow_html=True,
        )
else:
    st.info("👆 Upload a histopathology patch image above to get started.")

    with st.expander("About this model"):
        st.markdown("""
        - **Architecture**: ResNet50 pre-trained on ImageNet, fine-tuned for binary classification
        - **Dataset**: BreakHis (BreaKHis v1) - 7,909 images across 8 tumour subtypes
        - **Training split**: 70% train / 15% val / 15% test (slide-level, no leakage)
        - **Classes**: Benign (adenosis, fibroadenoma, phyllodes tumour, tubular adenoma)
          and Malignant (ductal carcinoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma)
        - **Input**: 224×224 RGB patch, normalised to ImageNet statistics
        """)

with st.sidebar:
    st.markdown("### API Status")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if health.status_code == 200:
            st.success("Backend online")
        else:
            st.warning(f"Backend returned {health.status_code}")
    except Exception:
        st.error("Backend offline")

    st.markdown(f"**Endpoint:** `{BACKEND_URL}`")
    st.divider()
    st.markdown("Built with FastAPI + Streamlit + PyTorch")
