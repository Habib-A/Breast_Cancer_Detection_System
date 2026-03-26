import streamlit as st
import requests
import os
from PIL import Image
from io import BytesIO

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Histopathology Analysis",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean medical aesthetic */
    body { background-color: #f7f2ff; }
    .main { background-color: #f7f2ff; }
    .stApp { font-family: 'Source Sans Pro', sans-serif; color: white; }

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
st.title("🔬 Breast Histopathology Analysis")
st.markdown(
    "Upload one or more histopathology patch images to analyze and classify them as **benign** or **malignant** "
    "using a fine-tuned ResNet50 model trained on the BreakHis dataset."
)
st.divider()

# ── Upload options (images vs folder) + 2-column layout ─────────────────────
upload_mode = st.radio(
    "Upload mode",
    ["Upload images", "Upload folder"],
    horizontal=True,
)

uploader_label = "Upload images (PNG or JPEG)"
uploader_help = "Select one or multiple image files. Ideally 224×224 patches."
if upload_mode == "Upload folder":
    uploader_label = "Upload folder (select multiple files)"
    uploader_help = "Choose multiple images from your folder (Streamlit folder upload uses multi-select)."

uploaded_files = st.file_uploader(
    uploader_label,
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help=uploader_help,
)

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("**Uploaded images (preview)**")
    if uploaded_files:
        preview_cols = st.columns(3, gap="small")
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(BytesIO(uploaded_file.getvalue()))
            with preview_cols[idx % 3]:
                st.image(image, use_column_width=True)
                st.caption(uploaded_file.name)
    else:
        st.info("Upload images or a folder to see previews here.")

with right_col:
    st.markdown("**Prediction results**")
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} image(s) uploaded")

        with st.spinner("Running inference..."):
            results = []
            for uploaded_file in uploaded_files:
                try:
                    response = requests.post(
                        PREDICT_ENDPOINT,
                        files={"file": (uploaded_file.name,
                                        uploaded_file.getvalue(),
                                        uploaded_file.type)},
                        timeout=120,
                    )
                    response.raise_for_status()
                    result = response.json()
                    results.append(
                        {
                            "filename": uploaded_file.name,
                            **result,
                        }
                    )
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot reach the backend API. "
                        f"Expected URL: `{BACKEND_URL}`. "
                        "On Railway, ensure FastAPI is healthy and the model weights are available "
                        "(`Model/best_model.pth` via volume, or `MODEL_DOWNLOAD_URL`)."
                    )
                    st.stop()
                except requests.exceptions.Timeout:
                    st.error("Request timed out. If this is the first request after deploy, try again in a moment.")
                    st.stop()
                except requests.exceptions.HTTPError as e:
                    st.error(f"API error for `{uploaded_file.name}`: {e.response.status_code} - {e.response.text}")
                    st.stop()
                except Exception as e:
                    st.error(f"Unexpected error for `{uploaded_file.name}`: {e}")
                    st.stop()

        for result in results:
            prediction = result["prediction"]
            confidence = result["confidence"]
            probs = result["probabilities"]

            card_class = "result-malignant" if prediction == "malignant" else "result-benign"
            icon = "⚠️" if prediction == "malignant" else "✅"

            st.markdown(
                f"""
                <div class="result-card {card_class}">
                    <div class="result-title">{icon} {prediction.capitalize()}</div>
                    <div class="confidence-text">
                        Confidence: <strong>{confidence}%</strong>
                    </div>
                    <div class="confidence-text" style="opacity:0.75;">
                        File: {result['filename']}
                    </div>
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
            st.divider()

        st.markdown(
            '<div class="disclaimer">⚕️ <strong>For research purposes only.</strong> '
            'This tool is not a medical device and must not be used for clinical diagnosis. '
            'Always consult a qualified pathologist.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("👆 Upload histopathology patch images above to get started.")

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
