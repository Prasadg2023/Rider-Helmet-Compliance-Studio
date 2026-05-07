from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO


st.set_page_config(page_title="Rider Helmet Compliance Studio", layout="wide")

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs" / "detect"
SAMPLE_DIR = ROOT / "JPEGImages"
MODEL_PATH = ROOT / "best.pt"
CLASS_LABELS = {0: "Rider With Helmet", 1: "Rider Without Helmet"}


def display_model_name(model_path: Path) -> str:
    if model_path == MODEL_PATH:
        return "bundled_best"
    try:
        return model_path.parents[1].name
    except IndexError:
        return model_path.stem


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-top: #07111f;
            --bg-bottom: #0f1b2e;
            --panel: rgba(13, 23, 40, 0.88);
            --panel-strong: rgba(17, 28, 46, 0.96);
            --border: rgba(148, 163, 184, 0.14);
            --text-main: #e6eef8;
            --text-soft: #9db0c8;
            --success-bg: rgba(20, 83, 45, 0.48);
            --warn-bg: rgba(120, 53, 15, 0.42);
            --danger-bg: rgba(127, 29, 29, 0.45);
            --shadow: 0 24px 50px rgba(2, 6, 23, 0.36);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(34, 197, 94, 0.18), transparent 26%),
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.16), transparent 24%),
                linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
            color: var(--text-main);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #08111f 0%, #0b1728 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.10);
        }
        [data-testid="stSidebar"] * {
            color: var(--text-main);
        }
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        .stTextInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stSlider,
        .stRadio,
        .stFileUploader,
        .stButton button {
            color: var(--text-main) !important;
        }
        .stButton button {
            background: linear-gradient(135deg, #163047, #1f4b3b);
            border: 1px solid rgba(74, 222, 128, 0.24);
            border-radius: 14px;
        }
        .stButton button:hover {
            border-color: rgba(74, 222, 128, 0.45);
            color: white !important;
        }
        .hero {
            padding: 1.8rem 2rem;
            border-radius: 28px;
            color: var(--text-main);
            background:
                linear-gradient(135deg, rgba(10,18,31,0.96), rgba(16,31,51,0.94)),
                radial-gradient(circle at right top, rgba(56,189,248,0.14), transparent 30%);
            border: 1px solid rgba(74, 222, 128, 0.14);
            box-shadow: var(--shadow);
            margin-bottom: 1.1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.35rem;
            letter-spacing: -0.03em;
            color: #f8fafc;
        }
        .hero p {
            margin: 0.55rem 0 0;
            color: var(--text-soft);
            font-size: 1.02rem;
            max-width: 52rem;
            line-height: 1.55;
        }
        .info-card {
            padding: 1.05rem 1.15rem;
            border-radius: 20px;
            background: var(--panel);
            border: 1px solid var(--border);
            box-shadow: 0 14px 28px rgba(2, 6, 23, 0.28);
            min-height: 114px;
            backdrop-filter: blur(10px);
        }
        .info-card .label {
            color: var(--text-soft);
            font-size: 0.9rem;
            margin-bottom: 0.35rem;
        }
        .info-card .value {
            color: var(--text-main);
            font-size: 1.75rem;
            font-weight: 700;
        }
        .info-card .sub {
            color: var(--text-soft);
            font-size: 0.9rem;
            margin-top: 0.35rem;
            line-height: 1.45;
        }
        .section-shell {
            margin-top: 1rem;
            padding: 1rem 1.1rem;
            background: var(--panel-strong);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: 0 14px 30px rgba(2, 6, 23, 0.24);
        }
        .section-shell h3 {
            margin-top: 0.1rem;
            color: var(--text-main);
        }
        [data-testid="stMetric"] {
            background: var(--panel-strong);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.24);
        }
        [data-testid="stMetricLabel"] {
            color: var(--text-soft);
        }
        [data-testid="stMetricValue"] {
            color: #f8fafc;
        }
        [data-testid="stImage"] img {
            border-radius: 20px;
            border: 1px solid var(--border);
            box-shadow: 0 14px 32px rgba(2, 6, 23, 0.28);
        }
        .stAlert {
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.10);
        }
        [data-testid="stNotificationContentSuccess"] {
            background: var(--success-bg);
        }
        [data-testid="stNotificationContentWarning"] {
            background: var(--warn-bg);
        }
        [data-testid="stNotificationContentError"] {
            background: var(--danger-bg);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 18px;
            overflow: hidden;
        }
        [data-testid="stDataFrame"] * {
            color: var(--text-main) !important;
            background: transparent !important;
        }
        .pill-row {
            display: flex;
            gap: 0.65rem;
            flex-wrap: wrap;
            margin-top: 0.9rem;
        }
        .pill {
            padding: 0.5rem 0.85rem;
            border-radius: 999px;
            background: rgba(19, 34, 56, 0.92);
            border: 1px solid rgba(74, 222, 128, 0.18);
            color: #dff7e8;
            font-size: 0.88rem;
            font-weight: 600;
        }
        .pill.cool {
            border-color: rgba(56, 189, 248, 0.22);
            color: #d8f1ff;
        }
        .block-label {
            color: var(--text-soft);
            font-size: 0.9rem;
            margin-bottom: 0.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def available_models() -> dict[str, Path]:
    if MODEL_PATH.exists():
        return {"bundled_best.pt": MODEL_PATH}
    candidates = sorted(RUNS_DIR.glob("helmet_model*/weights/best.pt"))
    return {f"{path.parents[1].name}  |  {path.stat().st_mtime_ns}": path for path in candidates}


def available_samples(limit: int = 40) -> list[Path]:
    if not SAMPLE_DIR.exists():
        return []
    return sorted(
        [path for path in SAMPLE_DIR.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )[:limit]


@st.cache_resource(show_spinner=False)
def load_model(model_path_str: str) -> YOLO:
    return YOLO(model_path_str)


def run_prediction(model: YOLO, image: Image.Image, conf: float, iou: float):
    image_np = np.array(image.convert("RGB"))
    result = model.predict(image_np, conf=conf, iou=iou, imgsz=640, augment=True, verbose=False)[0]
    used_fallback = False

    if result.boxes is None or len(result.boxes) == 0:
        result = model.predict(
            image_np,
            conf=min(conf, 0.20),
            iou=iou,
            imgsz=736,
            augment=True,
            verbose=False,
        )[0]
        used_fallback = True

    return result, result.plot(), used_fallback

def summarize_predictions(result) -> tuple[int, int, list[dict[str, str]]]:
    boxes = result.boxes
    if boxes is None or boxes.cls is None or len(boxes) == 0:
        return 0, 0, []

    class_ids = boxes.cls.int().tolist()
    confidences = boxes.conf.tolist()
    coords = boxes.xyxy.tolist()
    with_helmet = sum(1 for class_id in class_ids if class_id == 0)
    without_helmet = sum(1 for class_id in class_ids if class_id == 1)

    rows = []
    for idx, (class_id, confidence, xyxy) in enumerate(zip(class_ids, confidences, coords), start=1):
        x1, y1, x2, y2 = [int(value) for value in xyxy]
        rows.append(
            {
                "Detection": str(idx),
                "Class": CLASS_LABELS.get(class_id, str(class_id)),
                "Confidence": f"{confidence:.2%}",
                "Box": f"({x1}, {y1}) to ({x2}, {y2})",
            }
        )
    return with_helmet, without_helmet, rows


def info_card(label: str, value: str, sub: str) -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_styles()
st.markdown(
    """
    <div class="hero">
        <h1>Rider Helmet Compliance Studio</h1>
        <p>Inspect rider-scene helmet compliance predictions with a cleaner dashboard, sample-image testing, and quick model switching.</p>
        <div class="pill-row">
            <div class="pill">Rider-scene safety detection</div>
            <div class="pill cool">CPU-friendly model workflow</div>
            <div class="pill">Upload, sample, and webcam modes</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

models = available_models()
if not models:
    st.error("No trained model found. Train the model first so a best.pt file exists in runs/detect.")
    st.stop()

sorted_model_items = sorted(models.items(), key=lambda item: item[1].stat().st_mtime, reverse=True)
model_labels = [label for label, _ in sorted_model_items]
model_lookup = {label: path for label, path in sorted_model_items}

st.sidebar.title("Control Panel")
selected_label = st.sidebar.selectbox("Model", model_labels, index=0)
selected_model_path = model_lookup[selected_label]
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4, 0.05)
iou = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)
show_table = st.sidebar.toggle("Show Detection Table", value=True)
sample_paths = available_samples()
input_modes = ["Upload", "Webcam"]
if sample_paths:
    input_modes.insert(1, "Sample")
input_mode = st.sidebar.radio("Input Mode", input_modes, index=0)

selected_sample = "None"
if input_mode == "Sample":
    selected_sample = st.sidebar.selectbox("Quick Sample", ["None"] + [path.name for path in sample_paths], index=0)

model = load_model(str(selected_model_path))

uploaded_file = None
camera_file = None
if input_mode == "Upload":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
elif input_mode == "Webcam":
    camera_file = st.camera_input("Capture From Webcam")

selected_image = None
selected_name = None
if uploaded_file is not None:
    selected_image = Image.open(uploaded_file).convert("RGB")
    selected_name = uploaded_file.name
elif camera_file is not None:
    selected_image = Image.open(camera_file).convert("RGB")
    selected_name = "Webcam Capture"
elif selected_sample != "None":
    sample_path = next(path for path in sample_paths if path.name == selected_sample)
    selected_image = Image.open(sample_path).convert("RGB")
    selected_name = sample_path.name

top_left, top_mid, top_right = st.columns(3)
with top_left:
    info_card("Active Model", display_model_name(selected_model_path), "Bundled model is used for cloud deployment.")
with top_mid:
    info_card("Confidence", f"{confidence:.2f}", "Raise it to reduce weak rider detections.")
with top_right:
    info_card("Image Source", selected_name or "Waiting", "Upload a file or use a sample image.")

if selected_image is None:
    if input_mode == "Webcam":
        st.info("Allow camera access and capture a frame to start detection.")
    elif input_mode == "Sample":
        st.info("Choose a sample image from the sidebar to start detection.")
    else:
        st.info("Upload an image to start detection.")
    st.stop()

with st.spinner("Running detection..."):
    result, annotated, used_fallback = run_prediction(model, selected_image, confidence, iou)

with_helmet, without_helmet, detection_rows = summarize_predictions(result)
total_count = with_helmet + without_helmet

status_col, metric_col1, metric_col2, metric_col3 = st.columns([1.3, 1, 1, 1])
with status_col:
    if without_helmet > 0:
        st.warning("Safety alert: one or more riders without helmets were detected.")
    elif total_count > 0:
        st.success("All detected riders appear to be wearing helmets.")
    else:
        st.info("No riders were detected in this image.")
with metric_col1:
    st.metric("Total Detections", total_count)
with metric_col2:
    st.metric("Rider With Helmet", with_helmet)
with metric_col3:
    st.metric("Rider Without Helmet", without_helmet)

if used_fallback:
    st.caption("Fallback detection mode was used with a lower confidence threshold to reduce missed riders.")

image_col1, image_col2 = st.columns(2)
with image_col1:
    st.markdown('<div class="block-label">Original Frame</div>', unsafe_allow_html=True)
    st.image(selected_image, use_container_width=True)
with image_col2:
    st.markdown('<div class="block-label">Model Prediction</div>', unsafe_allow_html=True)
    st.image(annotated, use_container_width=True)

if detection_rows and show_table:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.subheader("Detection Details")
    st.dataframe(detection_rows, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
elif total_count == 0:
    st.caption("No detection rows to display for this image.")

st.markdown('<div class="section-shell">', unsafe_allow_html=True)
st.subheader("Session Notes")
st.write(
    f"Model file: `{selected_model_path}`. "
    f"This cloud app uses the bundled trained model and is intended for rider or head scenes, not standalone helmet product photos. "
    f"Use the sidebar to compare sample images, tune thresholds, and inspect how stable the predictions feel."
)
st.markdown("</div>", unsafe_allow_html=True)
