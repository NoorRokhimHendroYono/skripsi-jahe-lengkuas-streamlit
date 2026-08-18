import os
from pathlib import Path

import streamlit as st
import tensorflow as tf


# ============================================================
# STREAMLIT — STAGE 1
# Environment Check + Load 3 Final Models
# ============================================================

st.set_page_config(
    page_title="Klasifikasi Jahe & Lengkuas",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 Klasifikasi Jahe dan Lengkuas")
st.caption("Deep Learning Image Classification — Stage 1: Model Loading")


# ------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------

COLAB_ROOT = Path("/content/drive/MyDrive/SKRIPSI_AI")

# Untuk testing di Google Colab.
# Untuk deployment nanti, kita akan ubah ke struktur project
# relatif agar dapat dijalankan dari GitHub/hosting.
if COLAB_ROOT.exists():
    ROOT_DIR = COLAB_ROOT
else:
    ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "Saved_Model"

MODEL_PATHS = {
    "Baseline CNN": MODEL_DIR / "Baseline_CNN" / "baseline_cnn_final.keras",
    "MobileNetV2": MODEL_DIR / "MobileNetV2" / "mobilenetv2_final.keras",
    "EfficientNetB0": MODEL_DIR / "EfficientNetB0" / "efficientnetb0_final.keras",
}


# ------------------------------------------------------------
# ENVIRONMENT CHECK
# ------------------------------------------------------------

st.subheader("1. Environment Check")

tf_version = tf.__version__
gpus = tf.config.list_physical_devices("GPU")

env_col1, env_col2 = st.columns(2)

with env_col1:
    st.write("**TensorFlow:**", tf_version)

with env_col2:
    st.write(
        "**GPU:**",
        "Tersedia" if gpus else "Tidak tersedia"
    )

if gpus:
    st.success(f"GPU terdeteksi: {gpus}")
else:
    st.info(
        "GPU tidak terdeteksi. Untuk tahap Streamlit prediction, "
        "CPU masih dapat digunakan; GPU akan dipertimbangkan saat deployment."
    )


# ------------------------------------------------------------
# MODEL PATH CHECK
# ------------------------------------------------------------

st.subheader("2. Model File Check")

path_rows = []

for model_name, model_path in MODEL_PATHS.items():
    path_rows.append(
        {
            "Model": model_name,
            "File": model_path.name,
            "Status": "FOUND" if model_path.exists() else "NOT FOUND",
            "Path": str(model_path),
        }
    )

st.dataframe(
    path_rows,
    use_container_width=True,
    hide_index=True,
)


missing_models = [
    model_name
    for model_name, model_path in MODEL_PATHS.items()
    if not model_path.exists()
]

if missing_models:
    st.error(
        "Model belum lengkap. File yang tidak ditemukan: "
        + ", ".join(missing_models)
    )
    st.stop()

st.success("Semua 3 final model ditemukan.")


# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------

st.subheader("3. Load Final Models")

@st.cache_resource
def load_models():
    loaded = {}

    for model_name, model_path in MODEL_PATHS.items():
        loaded[model_name] = tf.keras.models.load_model(
            model_path,
            compile=False,
        )

    return loaded


with st.spinner("Memuat 3 final model..."):
    models = load_models()


# ------------------------------------------------------------
# MODEL CHECK
# ------------------------------------------------------------

model_rows = []

for model_name, model in models.items():
    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)

    model_rows.append(
        {
            "Model": model_name,
            "Input Shape": str(input_shape),
            "Output Shape": str(output_shape),
            "Status": "PASS",
        }
    )

st.dataframe(
    model_rows,
    use_container_width=True,
    hide_index=True,
)


# ------------------------------------------------------------
# FINAL CHECK
# ------------------------------------------------------------

st.subheader("4. Final Check")

all_loaded = len(models) == 3

if all_loaded:
    st.success(
        "STREAMLIT STAGE 1 PASS — "
        "Baseline CNN, MobileNetV2, dan EfficientNetB0 berhasil dimuat."
    )
else:
    st.error("STREAMLIT STAGE 1 FAIL.")

st.divider()

st.info(
    "Tahap berikutnya: Prediction Engine. "
    "Kita akan menambahkan upload citra, preprocessing 224×224 RGB, "
    "prediksi ketiga model, dan confidence."
)
