import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# STREAMLIT — STAGE 2
# Prediction Engine
# Upload Image + Preprocessing + 3 Model Prediction
# ============================================================

st.set_page_config(
    page_title="Klasifikasi Jahe & Lengkuas",
    page_icon="🌿",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

st.title("🌿 Klasifikasi Jahe dan Lengkuas")
st.caption(
    "Deep Learning Image Classification — "
    "Stage 2: Prediction Engine"
)


# ------------------------------------------------------------
# CLASS MAPPING
# ------------------------------------------------------------

CLASS_NAMES = [
    "Jahe",
    "Lengkuas",
]

IMAGE_SIZE = (224, 224)


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

MODEL_DIR = ROOT_DIR / "models"


# ------------------------------------------------------------
# MODEL PATHS
# ------------------------------------------------------------

MODEL_PATHS = {
    "Baseline CNN":
        MODEL_DIR / "baseline_cnn_final.keras",

    "MobileNetV2":
        MODEL_DIR / "mobilenetv2_final.keras",

    "EfficientNetB0":
        MODEL_DIR / "efficientnetb0_final.keras",
}


# ============================================================
# STAGE 1 — ENVIRONMENT CHECK
# ============================================================

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
    st.success(
        f"GPU terdeteksi: {len(gpus)} device."
    )
else:
    st.info(
        "GPU tidak terdeteksi. "
        "Prediction tetap dapat dijalankan menggunakan CPU."
    )


# ============================================================
# STAGE 1 — MODEL FILE CHECK
# ============================================================

st.subheader("2. Model File Check")

path_rows = []

for model_name, model_path in MODEL_PATHS.items():

    path_rows.append(
        {
            "Model": model_name,
            "File": model_path.name,
            "Status": (
                "FOUND"
                if model_path.exists()
                else "NOT FOUND"
            ),
            "Path": str(model_path),
        }
    )

st.dataframe(
    path_rows,
    width="stretch",
    hide_index=True,
)


missing_models = [
    model_name
    for model_name, model_path in MODEL_PATHS.items()
    if not model_path.exists()
]

if missing_models:

    st.error(
        "Model belum lengkap. "
        "File yang tidak ditemukan: "
        + ", ".join(missing_models)
    )

    st.stop()

st.success(
    "Semua 3 final model ditemukan."
)


# ============================================================
# STAGE 1 — LOAD MODELS
# ============================================================

st.subheader("3. Load Final Models")


@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, model_path in MODEL_PATHS.items():

        loaded_models[model_name] = (
            tf.keras.models.load_model(
                model_path,
                compile=False,
            )
        )

    return loaded_models


with st.spinner(
    "Memuat Baseline CNN, MobileNetV2, "
    "dan EfficientNetB0..."
):

    models = load_models()


# ============================================================
# STAGE 1 — MODEL VALIDATION
# ============================================================

model_rows = []

for model_name, model in models.items():

    model_rows.append(
        {
            "Model": model_name,
            "Input Shape": str(
                tuple(model.input_shape)
            ),
            "Output Shape": str(
                tuple(model.output_shape)
            ),
            "Status": "PASS",
        }
    )


st.dataframe(
    model_rows,
    width="stretch",
    hide_index=True,
)


# ============================================================
# STAGE 1 FINAL CHECK
# ============================================================

all_loaded = len(models) == 3

if not all_loaded:

    st.error(
        "STAGE 1 FAIL — "
        "Tidak semua model berhasil dimuat."
    )

    st.stop()


st.success(
    "STAGE 1 PASS — "
    "Ketiga final model berhasil dimuat."
)


# ============================================================
# STAGE 2 — IMAGE UPLOAD
# ============================================================

st.divider()

st.header("4. Upload Citra")

st.write(
    "Upload satu citra jahe atau lengkuas "
    "untuk dilakukan klasifikasi oleh ketiga model."
)


uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_image(image):

    # Pastikan RGB
    image = image.convert("RGB")

    # Resize sesuai input model
    image = image.resize(
        IMAGE_SIZE
    )

    # Convert ke NumPy
    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    # Normalisasi 0–1
    image_array = image_array / 255.0

    # Tambahkan batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return image_array


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_model(model, input_array):

    prediction = model.predict(
        input_array,
        verbose=0,
    )

    probabilities = prediction[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        probabilities,
    )


# ============================================================
# PREDICTION ENGINE
# ============================================================

if uploaded_file is not None:

    st.divider()

    st.header("5. Prediction Engine")

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    )

    image = image.convert("RGB")

    # --------------------------------------------------------
    # DISPLAY ORIGINAL IMAGE
    # --------------------------------------------------------

    image_col1, image_col2 = st.columns(
        [1, 1]
    )

    with image_col1:

        st.subheader(
            "Citra Input"
        )

        st.image(
            image,
            caption=(
                f"{uploaded_file.name} "
                f"— {image.size[0]}×{image.size[1]} px"
            ),
            width="stretch",
        )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    input_array = preprocess_image(
        image
    )

    with image_col2:

        st.subheader(
            "Preprocessing"
        )

        st.write(
            "**Color:** RGB"
        )

        st.write(
            "**Ukuran input:** "
            "224 × 224 × 3"
        )

        st.write(
            "**Normalisasi:** "
            "0–1"
        )

        st.write(
            "**Batch shape:** "
            f"{input_array.shape}"
        )


    # ========================================================
    # PREDICT ALL MODELS
    # ========================================================

    results = []

    with st.spinner(
        "Menjalankan prediksi "
        "ketiga model..."
    ):

        for model_name, model in models.items():

            (
                predicted_class,
                confidence,
                probabilities,
            ) = predict_model(
                model,
                input_array,
            )

            results.append(
                {
                    "Model": model_name,
                    "Prediksi": predicted_class,
                    "Confidence": confidence,
                }
            )


    results_df = pd.DataFrame(
        results
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.subheader(
        "6. Hasil Prediksi"
    )

    display_df = results_df.copy()

    display_df["Confidence"] = (
        display_df["Confidence"] * 100
    ).map(
        lambda x: f"{x:.2f}%"
    )


    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )


    # ========================================================
    # INDIVIDUAL MODEL RESULT
    # ========================================================

    st.subheader(
        "7. Detail Prediksi Model"
    )

    result_columns = st.columns(3)

    for column, result in zip(
        result_columns,
        results,
    ):

        with column:

            st.markdown(
                f"### {result['Model']}"
            )

            st.metric(
                "Prediksi",
                result["Prediksi"],
            )

            st.metric(
                "Confidence",
                f"{result['Confidence'] * 100:.2f}%",
            )


    # ========================================================
    # AGREEMENT CHECK
    # ========================================================

    st.subheader(
        "8. Konsistensi Prediksi"
    )

    predicted_classes = [
        result["Prediksi"]
        for result in results
    ]

    if len(set(predicted_classes)) == 1:

        agreed_class = predicted_classes[0]

        st.success(
            "Ketiga model memberikan "
            f"prediksi yang sama: **{agreed_class}**"
        )

    else:

        st.warning(
            "Prediksi ketiga model tidak sepenuhnya sama."
        )

        st.write(
            "Hasil:",
            ", ".join(predicted_classes),
        )


    # ========================================================
    # STAGE 2 FINAL CHECK
    # ========================================================

    st.divider()

    st.subheader(
        "9. Stage 2 Final Check"
    )

    st.success(
        "STREAMLIT STAGE 2 PASS — "
        "Citra berhasil diproses dan "
        "diprediksi oleh 3 final model."
    )

else:

    st.divider()

    st.info(
        "Silakan upload satu citra untuk "
        "memulai prediction engine."
    )