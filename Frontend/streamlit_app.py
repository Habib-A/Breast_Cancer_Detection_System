import base64
import streamlit as st
import requests
import os
from PIL import Image
from io import BytesIO
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Histopathology Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean medical aesthetic */
    html {
        scroll-behavior: smooth;
    }
    body { background-color: #f7f2ff; }
    .main { background-color: #f7f2ff; }
    .stApp { font-family: 'Source Sans Pro', sans-serif; color: #1f1f1f; }

    /*
      Streamlit’s top bar (Deploy / menu) overlaps the top of the scrollable area.
      Extra padding-top keeps the custom header fully visible below it.
    */
    div.block-container {
        max-width: 2200px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        /* Clear the fixed Streamlit header (~56px) + small gap */
        padding-top: 4.25rem !important;
    }

    /* Header band: no negative margins (they caused overflow clipping in some layouts) */
    .page-header {
        background: linear-gradient(180deg, #4a3d6b 0%, #3d3358 100%);
        color: #f5f2ff;
        text-align: center;
        padding: 0.55rem 1rem 0.65rem 1rem;
        margin: 0 0 0.75rem 0;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12);
    }
    .page-header-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0 0 0.2rem 0;
        line-height: 1.2;
        color: #ffffff;
        letter-spacing: 0.01em;
    }
    .page-header-sub {
        font-size: 0.85rem;
        margin: 0;
        line-height: 1.35;
        color: #e8e0ff;
        font-weight: 400;
    }
    .page-header-sub strong {
        color: #ffffff;
        font-weight: 600;
    }

    /* Section title cards inside the two columns */
    .panel-title-card {
        background: linear-gradient(180deg, #4a3d6b 0%, #3d3358 100%);
        color: #ffffff;
        border-radius: 8px;
        padding: 0.35rem 0.6rem;
        margin: 0.25rem 0 0.55rem 0;
        font-size: 0.9rem;
        font-weight: 700;
        line-height: 1.2;
    }

    /*
      Main two panels (upload + prediction): slightly darker than page (#f7f2ff),
      much lighter than header (#3d3358 / #4a3d6b). Uses Streamlit bordered containers.
    */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(180deg, #ebe3f7 0%, #e4daf2 100%) !important;
        border-color: #cfc3e3 !important;
        border-radius: 12px !important;
    }

    /* Compact the upload controls row (radio + uploader) */
    .upload-controls-label {
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem 0 0.25rem 0;
    }
    div[data-testid="stRadio"] label, div[data-testid="stRadio"] p {
        font-size: 0.82rem !important;
    }
    div[data-testid="stRadio"] { margin-bottom: 0.25rem !important; }
    div[data-testid="stFileUploader"] { margin-top: 0.1rem !important; margin-bottom: 0.25rem !important; }
    div[data-testid="stFileUploader"] label { font-size: 0.82rem !important; }
    /* Hide Streamlit's selected-file list under the uploader (structure varies by version) */
    div[data-testid="stFileUploaderFileList"],
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileList"],
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
    div[data-testid="stFileUploader"] ul,
    div[data-testid="stFileUploader"] li {
        display: none !important;
    }
    /* Everything after the dropzone is the chip/list of uploaded files */
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] ~ *,
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] ~ * {
        display: none !important;
    }
    /* Reduce the uploader dropzone padding a bit */
    div[data-testid="stFileUploaderDropzone"] {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }

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

# ── Header (compact, centered, darker band) ───────────────────────────────────
st.markdown(
    """
<div class="page-header">
  <div class="page-header-title">🔬 Breast Histopathology Analysis</div>
  <p class="page-header-sub">
    Upload one or more histopathology patch images to analyze and classify them as
    <strong>benign</strong> or <strong>malignant</strong>
    using a fine-tuned ResNet50 model.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ── 2-column layout (left: upload+preview, right: results) ──────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    with st.container(border=True):
        # compact controls row INSIDE the left panel
        controls_left, controls_right = st.columns([0.55, 1.45], gap="medium")

        with controls_left:
            st.markdown('<div class="upload-controls-label">Upload mode</div>', unsafe_allow_html=True)
            upload_mode = st.radio(
                label="Upload mode",
                options=["Images", "Folder"],
                horizontal=True,
                label_visibility="collapsed",
            )

        uploader_label = "Upload images (PNG/JPG)"
        uploader_help = "Select one or multiple image files."
        if upload_mode == "Folder":
            uploader_label = "Upload folder (select multiple files)"
            uploader_help = "Select multiple images from a folder (multi-select)."

        with controls_right:
            st.markdown(f'<div class="upload-controls-label">{uploader_label}</div>', unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                label=uploader_label,
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                help=uploader_help,
                label_visibility="collapsed",
            )

        st.markdown('<div class="panel-title-card">Uploaded images (preview)</div>', unsafe_allow_html=True)
        if uploaded_files:
            preview_cols = st.columns(3, gap="small")
            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(BytesIO(uploaded_file.getvalue()))
                with preview_cols[idx % 3]:
                    st.image(image, width=180)
                    st.caption(uploaded_file.name)
        else:
            st.info("Upload images or a folder to see previews here.")

with right_col:
    with st.container(border=True):
        st.markdown('<div class="panel-title-card">Prediction results</div>', unsafe_allow_html=True)
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

            results_table = []
            benign_count = 0
            malignant_count = 0
            for result in results:
                prediction = result["prediction"]
                probs = result["probabilities"]
                if prediction == "benign":
                    benign_count += 1
                else:
                    malignant_count += 1

                results_table.append(
                    {
                        "File": result["filename"],
                        "Prediction": prediction,
                        "Confidence (%)": result["confidence"],
                        "Benign (%)": probs["benign"],
                        "Malignant (%)": probs["malignant"],
                    }
                )

            st.markdown(f"**Benign:** {benign_count}  |  **Malignant:** {malignant_count}")

            # Matplotlib pie (Altair + Streamlit Vega-Lite path is brittle without pandas / across versions).
            total_n = benign_count + malignant_count
            if total_n == 0:
                st.caption("No predictions to chart.")
            else:
                # True half-donut: only 180° arc via Wedge (pie + ylim often redraws as a full ring).
                fig, ax = plt.subplots(figsize=(2.0, 1.05), dpi=110)
                fig.patch.set_facecolor("none")
                fig.patch.set_alpha(0)
                ax.set_facecolor("none")
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                sizes = [benign_count, malignant_count]
                colors = ["#38a169", "#e53e3e"]
                outer_r = 1.0
                inner_r = 0.48
                radial_w = outer_r - inner_r

                cursor_deg = 0.0
                for i, val in enumerate(sizes):
                    span = (val / total_n) * 180.0
                    if span > 0:
                        ax.add_patch(
                            Wedge(
                                (0, 0),
                                outer_r,
                                cursor_deg,
                                cursor_deg + span,
                                width=radial_w,
                                facecolor=colors[i],
                                edgecolor="white",
                                linewidth=0.6,
                            )
                        )
                        mid = np.radians(cursor_deg + span / 2.0)
                        r_txt = (inner_r + outer_r) / 2.0
                        ax.text(
                            r_txt * np.cos(mid),
                            r_txt * np.sin(mid),
                            str(int(val)),
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white",
                            fontweight="bold",
                        )
                    cursor_deg += span

                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(0.0, 1.05)
                ax.set_aspect("equal", adjustable="box")
                ax.axis("off")

                # Center without nested st.columns (right_col is already a column; limit is one level).
                buf = BytesIO()
                fig.savefig(
                    buf,
                    format="png",
                    dpi=110,
                    bbox_inches="tight",
                    pad_inches=0.01,
                    transparent=True,
                )
                plt.close(fig)
                b64 = base64.b64encode(buf.getvalue()).decode()
                st.markdown(
                    f'<div style="text-align:center"><img src="data:image/png;base64,{b64}" '
                    'alt="Benign vs malignant counts" style="max-width:100%;height:auto;"/></div>',
                    unsafe_allow_html=True,
                )

            # No fixed height — a set height (e.g. 360px) reserves space and shows blank rows.
            st.dataframe(
                results_table,
                use_container_width=True,
                hide_index=True,
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
