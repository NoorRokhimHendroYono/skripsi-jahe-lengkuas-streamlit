import os
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
import pandas as pd
from PIL import Image


# ============================================================
# STREAMLIT
# STAGE 3
# Upload + Camera + Prediction + Grad-CAM
# ============================================================

st.set_page_config(
    page_title="Klasifikasi Jahe & Lengkuas",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 Klasifikasi Jahe dan Lengkuas")
st.caption(
    "Deep Learning Image Classification — "
    "Stage 3: Prediction + Grad-CAM"
)


# ============================================================
# CONSTANT
# ============================================================

CLASS_NAMES = [
    "Jahe",
    "Lengkuas",
]

IMAGE_SIZE = (224, 224)


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

MODEL_DIR = ROOT_DIR / "models"

MODEL_PATHS = {
    "Baseline CNN":
        MODEL_DIR / "baseline_cnn_final.keras",

    "MobileNetV2":
        MODEL_DIR / "mobilenetv2_final.keras",

    "EfficientNetB0":
        MODEL_DIR / "efficientnetb0_final.keras",
}


# ============================================================
# FINAL MODEL METRICS / GRAD-CAM CONFIG
# ============================================================

# CSV hasil evaluasi dapat ditempatkan di folder berikut agar
# accuracy final ikut tampil di Streamlit.
METRICS_DIR_CANDIDATES = [
    ROOT_DIR,
    ROOT_DIR / "Results" / "Model_Evaluation",
    ROOT_DIR / "results" / "Model_Evaluation",
    ROOT_DIR / "Model_Evaluation",
]

METRICS_FILE_CANDIDATES = [
    "MASTER_RESULTS_FINAL.csv",
    "master_results_final.csv",
]

# Target layer dikunci mengikuti hasil audit Notebook 08_GradCAM_Audit.
GRADCAM_TARGET_LAYERS = {
    "Baseline CNN": "conv2d_2",
    "MobileNetV2": "Conv_1",
    "EfficientNetB0": "top_conv",
}


# ============================================================
# ENVIRONMENT CHECK
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
        f"GPU terdeteksi: {gpus}"
    )
else:
    st.info(
        "GPU tidak terdeteksi. "
        "Prediction dan Grad-CAM tetap dapat dijalankan "
        "menggunakan CPU."
    )


# ============================================================
# MODEL METRICS
# ============================================================

def normalize_model_name(value):
    value = str(value).strip().lower()

    aliases = {
        "baseline cnn": "Baseline CNN",
        "baseline_cnn": "Baseline CNN",
        "cnn": "Baseline CNN",
        "mobilenetv2": "MobileNetV2",
        "mobilenetv2_1.00_224": "MobileNetV2",
        "efficientnetb0": "EfficientNetB0",
        "efficientnet b0": "EfficientNetB0",
    }

    return aliases.get(value, str(value).strip())


def find_metrics_csv():
    for directory in METRICS_DIR_CANDIDATES:
        for filename in METRICS_FILE_CANDIDATES:
            candidate = directory / filename

            if candidate.exists():
                return candidate

    return None


@st.cache_data
def load_accuracy_metrics():
    metrics_path = find_metrics_csv()

    if metrics_path is None:
        return {}, None, (
            "MASTER_RESULTS_FINAL.csv tidak ditemukan. "
            "Accuracy final tidak dapat diisi otomatis."
        )

    try:
        df = pd.read_csv(metrics_path)

        # Cari kolom nama model.
        model_col = None
        for column in df.columns:
            normalized = str(column).strip().lower()

            if normalized in {
                "model",
                "model_name",
                "modelname",
            }:
                model_col = column
                break

        # Cari kolom accuracy.
        accuracy_col = None
        for column in df.columns:
            normalized = str(column).strip().lower()

            if normalized in {
                "accuracy",
                "accuracy (%)",
                "val_accuracy",
                "test_accuracy",
                "mean_accuracy",
                "accuracy_mean",
            }:
                accuracy_col = column
                break

        if model_col is None or accuracy_col is None:
            return {}, metrics_path, (
                "Kolom model/accuracy pada "
                f"{metrics_path.name} tidak dikenali."
            )

        accuracy_metrics = {}

        for _, row in df.iterrows():
            model_name = normalize_model_name(
                row[model_col]
            )

            if model_name not in MODEL_PATHS:
                continue

            try:
                value = float(row[accuracy_col])

                # Jika accuracy tersimpan sebagai 0–1,
                # ubah ke persen hanya saat ditampilkan.
                accuracy_metrics[model_name] = value

            except (TypeError, ValueError):
                continue

        if not accuracy_metrics:
            return {}, metrics_path, (
                "Tidak ada accuracy final untuk 3 model "
                "yang berhasil dibaca."
            )

        return accuracy_metrics, metrics_path, None

    except Exception as error:
        return {}, metrics_path, (
            f"Gagal membaca metrics CSV: {error}"
        )


accuracy_metrics, accuracy_metrics_path, accuracy_metrics_warning = (
    load_accuracy_metrics()
)


# ============================================================
# MODEL FILE CHECK
# ============================================================

st.subheader("2. Model File Check")

path_rows = []

for model_name, model_path in MODEL_PATHS.items():

    path_rows.append(
        {
            "Model": model_name,
            "File": model_path.name,
            "Status":
                "FOUND"
                if model_path.exists()
                else "NOT FOUND",
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
    for model_name, model_path
    in MODEL_PATHS.items()
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
# LOAD MODELS
# ============================================================

st.subheader("3. Load Final Models")


@st.cache_resource
def load_models():

    loaded = {}

    for model_name, model_path in MODEL_PATHS.items():

        loaded[model_name] = (
            tf.keras.models.load_model(
                model_path,
                compile=False,
            )
        )

    return loaded


with st.spinner(
    "Memuat 3 final model..."
):

    models = load_models()


# ============================================================
# MODEL CHECK
# ============================================================

model_rows = []

for model_name, model in models.items():

    input_shape = tuple(
        model.input_shape
    )

    output_shape = tuple(
        model.output_shape
    )

    model_size_bytes = MODEL_PATHS[
        model_name
    ].stat().st_size

    model_size_mb = (
        model_size_bytes / (1024 ** 2)
    )

    accuracy_value = accuracy_metrics.get(
        model_name
    )

    accuracy_display = (
        f"{accuracy_value * 100:.2f}%"
        if accuracy_value is not None
        and accuracy_value <= 1
        else (
            f"{accuracy_value:.2f}%"
            if accuracy_value is not None
            else "N/A"
        )
    )

    model_rows.append(
        {
            "Model": model_name,
            "Input Shape": str(
                input_shape
            ),
            "Output Shape": str(
                output_shape
            ),
            "Accuracy": accuracy_display,
            "Model Size (MB)": f"{model_size_mb:.2f}",
            "Status": "PASS",
        }
    )


st.dataframe(
    model_rows,
    width="stretch",
    hide_index=True,
)

st.success(
    "STAGE 1 PASS — "
    "Ketiga final model berhasil dimuat."
)

if accuracy_metrics_warning:
    st.warning(
        accuracy_metrics_warning
    )
else:
    st.caption(
        "Accuracy final dibaca dari: "
        f"{accuracy_metrics_path}"
    )


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image_array = np.array(image)

    resized = cv2.resize(
        image_array,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    processed = resized.astype(np.float32)

    batch = np.expand_dims(
        processed,
        axis=0
    )

    return (
        image_array,
        resized,
        batch
    )

# def preprocess_image(image):

#     image = image.convert("RGB")

#     image_array = np.array(image)

#     resized = cv2.resize(
#         image_array,
#         IMAGE_SIZE,
#         interpolation=cv2.INTER_AREA,
#     )

#     normalized = (
#         resized.astype(np.float32) / 255.0
#     )

#     batch = np.expand_dims(
#         normalized,
#         axis=0,
#     )

#     return (
#         image_array,
#         resized,
#         batch,
#     )


# ============================================================
# PREDICTION
# ============================================================

def predict_models(
    models,
    image_batch,
):

    results = {}

    for model_name, model in models.items():

        # Waktu yang diukur hanya proses inference/predict,
        # tidak termasuk preprocessing dan Grad-CAM.
        start_time = time.perf_counter()

        prediction = model.predict(
            image_batch,
            verbose=0,
        )

        end_time = time.perf_counter()

        inference_time_ms = (
            end_time - start_time
        ) * 1000.0

        prediction = np.asarray(
            prediction
        )[0]

        predicted_index = int(
            np.argmax(prediction)
        )

        predicted_class = (
            CLASS_NAMES[
                predicted_index
            ]
        )

        confidence = float(
            prediction[
                predicted_index
            ]
        )

        model_size_bytes = MODEL_PATHS[
            model_name
        ].stat().st_size

        model_size_mb = (
            model_size_bytes / (1024 ** 2)
        )

        accuracy_value = accuracy_metrics.get(
            model_name
        )

        results[model_name] = {
            "prediction":
                predicted_class,
            "confidence":
                confidence,
            "probabilities":
                prediction,
            "class_index":
                predicted_index,
            "inference_time_ms":
                inference_time_ms,
            "model_size_mb":
                model_size_mb,
            "accuracy":
                accuracy_value,
        }

    return results


# ============================================================
# FIND GRAD-CAM TARGET LAYER
# ============================================================

def find_target_layer(model, target_name):
    """
    Mencari target layer secara rekursif.
    Mengembalikan:
        (container_model, target_layer)
    """
    try:
        return model, model.get_layer(target_name)
    except Exception:
        pass

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                target_layer = layer.get_layer(target_name)
                return layer, target_layer
            except Exception:
                continue

    return None, None


def find_last_conv_layer(model):

    # First try direct layers
    for layer in reversed(model.layers):
        try:
            output_shape = layer.output.shape

            if (
                len(output_shape) == 4
                and isinstance(
                    layer,
                    (
                        tf.keras.layers.Conv2D,
                        tf.keras.layers.SeparableConv2D,
                        tf.keras.layers.DepthwiseConv2D,
                    ),
                )
            ):
                return layer

        except Exception:
            continue

    # Search nested models
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                nested_layer = find_last_conv_layer(layer)
                if nested_layer is not None:
                    return nested_layer
            except Exception:
                pass

    return None


# ============================================================
# BUILD GRAD-CAM MODEL
# ============================================================

def build_gradcam_model(model, target_layer_name):
    """
    Membangun model Grad-CAM dengan feature map dan prediction
    dari forward-pass yang sama.

    Mendukung:
    - Baseline CNN dengan target layer langsung pada model.
    - MobileNetV2 / EfficientNetB0 dengan target layer
      berada di dalam nested backbone.
    """

    container_model, target_layer = find_target_layer(
        model,
        target_layer_name,
    )

    if target_layer is None:
        raise ValueError(
            f"Target layer `{target_layer_name}` tidak ditemukan."
        )

    # ========================================================
    # TARGET LAYER LANGSUNG PADA MODEL UTAMA
    # ========================================================
    if container_model is model:

        try:
            target_index = model.layers.index(
                target_layer
            )

            # Mulai dari output target layer.
            x = target_layer.output

            # Replay semua layer setelah target layer.
            for head_layer in model.layers[
                target_index + 1:
            ]:
                x = head_layer(
                    x,
                    training=False
                )

            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[
                    target_layer.output,
                    x,
                ],
                name=f"gradcam_{model.name}",
            )

            return (
                grad_model,
                target_layer.name,
            )

        except Exception as error:

            raise ValueError(
                "Grad-CAM Baseline CNN gagal dibangun "
                "dari target layer "
                f"`{target_layer_name}`: {error}"
            ) from error

    # ========================================================
    # TARGET LAYER DI DALAM NESTED BACKBONE
    # ========================================================
    try:

        backbone = container_model

        backbone_index = model.layers.index(
            backbone
        )

        backbone_input = backbone.input

        x = backbone.output

        # Replay classification head setelah backbone.
        for head_layer in model.layers[
            backbone_index + 1:
        ]:
            x = head_layer(
                x,
                training=False
            )

        grad_model = tf.keras.models.Model(
            inputs=backbone_input,
            outputs=[
                target_layer.output,
                x,
            ],
            name=f"gradcam_{model.name}",
        )

        return (
            grad_model,
            target_layer.name,
        )

    except Exception as error:

        raise ValueError(
            "Target layer berada pada nested backbone, "
            "tetapi graph Grad-CAM gagal dibangun: "
            f"{error}"
        ) from error

# def build_gradcam_model(model, target_layer_name):
#     """
#     Membangun model Grad-CAM dengan feature map dan prediction
#     yang berasal dari graph forward-pass yang sama.

#     Untuk Baseline CNN, target layer berada langsung pada model.

#     Untuk MobileNetV2/EfficientNetB0, target layer berada di dalam
#     nested backbone. Head model kemudian direplay dari output
#     backbone sehingga gradient tetap terhubung.
#     """

#     container_model, target_layer = find_target_layer(
#         model,
#         target_layer_name,
#     )

#     if target_layer is None:
#         raise ValueError(
#             f"Target layer `{target_layer_name}` tidak ditemukan."
#         )

#     # --------------------------------------------------------
#     # TARGET LAYER LANGSUNG PADA MODEL UTAMA
#     # --------------------------------------------------------
#     if container_model is model:
#         try:
#             grad_model = tf.keras.models.Model(
#                 inputs=model.inputs,
#                 outputs=[
#                     target_layer.output,
#                     model.output,
#                 ],
#                 name=f"gradcam_{model.name}",
#             )

#             return grad_model, target_layer.name

#         except Exception as error:
#             raise ValueError(
#                 "Target layer ditemukan pada model utama, "
#                 "tetapi Grad-CAM model gagal dibangun."
#             ) from error

#     # --------------------------------------------------------
#     # TARGET LAYER BERADA DI DALAM NESTED BACKBONE
#     # --------------------------------------------------------
#     try:
#         backbone = container_model

#         backbone_index = model.layers.index(
#             backbone
#         )

#         # Pastikan backbone memiliki input/output yang valid.
#         backbone_input = backbone.input
#         x = backbone.output

#         # Replay seluruh classification head setelah backbone.
#         for head_layer in model.layers[
#             backbone_index + 1:
#         ]:
#             x = head_layer(x)

#         grad_model = tf.keras.models.Model(
#             inputs=backbone_input,
#             outputs=[
#                 target_layer.output,
#                 x,
#             ],
#             name=f"gradcam_{model.name}",
#         )

#         return grad_model, target_layer.name

#     except Exception as error:
#         raise ValueError(
#             "Target layer berada pada nested backbone, "
#             "tetapi graph Grad-CAM gagal dibangun."
#         ) from error


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam_heatmap(
    image_batch,
    model,
    model_name,
    predicted_index,
):
    """
    Grad-CAM menggunakan target layer yang dikunci berdasarkan
    hasil audit Notebook 08_GradCAM_Audit.
    """

    expected_layer_name = (
        GRADCAM_TARGET_LAYERS.get(
            model_name
        )
    )

    if expected_layer_name is None:
        raise ValueError(
            f"Konfigurasi target layer tidak tersedia "
            f"untuk model `{model_name}`."
        )

    grad_model, layer_name = build_gradcam_model(
        model,
        expected_layer_name,
    )

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------
    with tf.GradientTape() as tape:

        conv_outputs, predictions = (
            grad_model(
                image_batch,
                training=False,
            )
        )

        class_channel = predictions[
            :, predicted_index
        ]

    # --------------------------------------------------------
    # GRADIENT
    # --------------------------------------------------------
    gradients = tape.gradient(
        class_channel,
        conv_outputs,
    )

    if gradients is None:
        raise ValueError(
            f"Gradient tidak tersedia untuk "
            f"model `{model_name}` pada layer "
            f"`{layer_name}`."
        )

    # --------------------------------------------------------
    # GLOBAL AVERAGE POOLING GRADIENT
    # --------------------------------------------------------
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    # --------------------------------------------------------
    # AMBIL FEATURE MAP PERTAMA
    # --------------------------------------------------------
    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    # --------------------------------------------------------
    # WEIGHTED COMBINATION
    # --------------------------------------------------------
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1,
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------
    heatmap = tf.maximum(
        heatmap,
        0,
    )

    # --------------------------------------------------------
    # NORMALISASI 0–1
    # --------------------------------------------------------
    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        tf.zeros_like(heatmap),
    )

    return (
        heatmap.numpy(),
        layer_name,
    )


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    original_image,
    heatmap,
):

    original = cv2.cvtColor(
        original_image,
        cv2.COLOR_RGB2BGR,
    )

    height, width = (
        original.shape[:2]
    )

    heatmap_resized = cv2.resize(
        heatmap,
        (width, height),
    )


    heatmap_uint8 = np.uint8(
        255 * heatmap_resized
    )


    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )


    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap_color,
        0.45,
        0,
    )


    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB,
    )

    return overlay


# ============================================================
# INPUT IMAGE
# ============================================================

st.divider()

st.subheader("4. Input Citra")

st.write(
    "Pilih salah satu metode input:"
)

input_tab1, input_tab2 = st.tabs(
    [
        "📁 Upload Gambar",
        "📷 Open Camera",
    ]
)


uploaded_file = None
camera_file = None


# ------------------------------------------------------------
# UPLOAD
# ------------------------------------------------------------

with input_tab1:

    uploaded_file = st.file_uploader(
        "Upload citra jahe atau lengkuas",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        help=(
            "Upload satu gambar untuk "
            "diklasifikasikan."
        ),
    )


# ------------------------------------------------------------
# CAMERA
# ------------------------------------------------------------

with input_tab2:

    camera_file = st.camera_input(
        "Ambil foto menggunakan kamera"
    )


# ============================================================
# SELECT INPUT
# ============================================================

image_source = (
    camera_file
    if camera_file is not None
    else uploaded_file
)


if image_source is None:

    st.info(
        "Silakan upload gambar atau "
        "ambil foto menggunakan kamera "
        "untuk memulai prediction."
    )

    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

try:

    image = Image.open(
        image_source
    ).convert("RGB")

except Exception as error:

    st.error(
        f"Gagal membaca gambar: {error}"
    )

    st.stop()


# ============================================================
# PREPROCESS
# ============================================================

(
    original_image,
    resized_image,
    image_batch,
) = preprocess_image(
    image
)


# ============================================================
# DISPLAY INPUT
# ============================================================

st.subheader(
    "5. Preprocessing"
)

input_col1, input_col2 = (
    st.columns(2)
)


with input_col1:

    st.markdown(
        "### Citra Input"
    )

    st.image(
        original_image,
        width="stretch",
    )


with input_col2:

    st.markdown(
        "### Informasi Preprocessing"
    )

    st.write(
        "**Color:** RGB"
    )

    st.write(
        "**Ukuran input:** "
        "224 × 224 × 3"
    )

    st.write(
        "**Normalisasi:** 0–1"
    )

    st.write(
        "**Batch shape:** "
        f"{image_batch.shape}"
    )


# ============================================================
# PREDICTION
# ============================================================

st.subheader(
    "6. Prediction Engine"
)

with st.spinner(
    "Melakukan prediksi dengan "
    "3 final model..."
):

    prediction_results = (
        predict_models(
            models,
            image_batch,
        )
    )


# ============================================================
# PREDICTION TABLE
# ============================================================

result_rows = []

for (
    model_name,
    result
) in prediction_results.items():

    result_rows.append(
        {
            "Model":
                model_name,

            "Prediksi":
                result["prediction"],

            "Confidence":
                f"{result['confidence'] * 100:.2f}%",

            "Accuracy":
                (
                    f"{result['accuracy'] * 100:.2f}%"
                    if result["accuracy"] is not None
                    and result["accuracy"] <= 1
                    else (
                        f"{result['accuracy']:.2f}%"
                        if result["accuracy"] is not None
                        else "N/A"
                    )
                ),

            "Inference Time (ms)":
                f"{result['inference_time_ms']:.2f}",

            "Model Size (MB)":
                f"{result['model_size_mb']:.2f}",
        }
    )


st.dataframe(
    result_rows,
    width="stretch",
    hide_index=True,
)


# ============================================================
# DETAIL PREDICTION
# ============================================================

st.subheader(
    "7. Detail Prediksi Model"
)

detail_cols = st.columns(3)


for (
    col,
    (model_name, result),
) in zip(
    detail_cols,
    prediction_results.items(),
):

    with col:

        st.markdown(
            f"### {model_name}"
        )

        st.write(
            "**Prediksi**"
        )

        st.markdown(
            f"## {result['prediction']}"
        )

        st.write(
            "**Confidence**"
        )

        st.markdown(
            f"## {result['confidence'] * 100:.2f}%"
        )

        st.write(
            "**Accuracy Model**"
        )

        if result["accuracy"] is not None:
            accuracy_display = (
                result["accuracy"] * 100
                if result["accuracy"] <= 1
                else result["accuracy"]
            )

            st.write(
                f"{accuracy_display:.2f}%"
            )
        else:
            st.write("N/A")

        st.write(
            "**Inference Time**"
        )

        st.write(
            f"{result['inference_time_ms']:.2f} ms"
        )

        st.write(
            "**Model Size**"
        )

        st.write(
            f"{result['model_size_mb']:.2f} MB"
        )


# ============================================================
# CONSISTENCY CHECK
# ============================================================

predicted_classes = [
    result["prediction"]
    for result
    in prediction_results.values()
]


st.subheader(
    "8. Konsistensi Prediksi"
)


if len(
    set(predicted_classes)
) == 1:

    st.success(
        "Ketiga model memberikan "
        "prediksi yang sama: "
        f"{predicted_classes[0]}"
    )

else:

    st.warning(
        "Ketiga model memberikan "
        "hasil prediksi yang berbeda."
    )


# ============================================================
# GRAD-CAM
# ============================================================

st.divider()

st.subheader(
    "9. Explainable AI — Grad-CAM"
)

st.write(
    "Grad-CAM digunakan untuk menunjukkan "
    "area citra yang paling berkontribusi "
    "terhadap keputusan model."
)


gradcam_results = {}


for (
    model_name,
    result,
) in prediction_results.items():

    with st.expander(
        f"🔍 Grad-CAM — {model_name}",
        expanded=True,
    ):

        try:

            heatmap, layer_name = (
                make_gradcam_heatmap(
                    image_batch,
                    models[model_name],
                    model_name,
                    result["class_index"],
                )
            )


            overlay = (
                create_gradcam_overlay(
                    original_image,
                    heatmap,
                )
            )


            gradcam_results[
                model_name
            ] = overlay


            st.write(
                f"**Target class:** "
                f"{result['prediction']}"
            )

            expected_layer = GRADCAM_TARGET_LAYERS.get(
                model_name
            )

            st.write(
                f"**Target layer:** "
                f"`{layer_name}`"
            )

            if expected_layer:
                if layer_name == expected_layer:
                    st.caption(
                        "Target layer sesuai "
                        "hasil audit Notebook 08."
                    )
                else:
                    st.warning(
                        "Target layer berbeda dari "
                        f"layer audit: `{expected_layer}`."
                    )


            grad_col1, grad_col2 = (
                st.columns(2)
            )


            with grad_col1:

                st.image(
                    original_image,
                    caption=(
                        "Citra Original"
                    ),
                    width="stretch",
                )


            with grad_col2:

                st.image(
                    overlay,
                    caption=(
                        "Grad-CAM Overlay"
                    ),
                    width="stretch",
                )


            st.success(
                "Grad-CAM berhasil dibuat."
            )


        except Exception as error:

            st.error(
                f"Grad-CAM gagal untuk "
                f"{model_name}: {error}"
            )


# ============================================================
# FINAL CHECK
# ============================================================

st.divider()

st.subheader(
    "10. Stage 3 Final Check"
)

st.write(
    "Komponen:"
)

final_metric_rows = []

for model_name, result in prediction_results.items():

    final_metric_rows.append(
        {
            "Model": model_name,
            "Accuracy":
                (
                    f"{result['accuracy'] * 100:.2f}%"
                    if result["accuracy"] is not None
                    and result["accuracy"] <= 1
                    else (
                        f"{result['accuracy']:.2f}%"
                        if result["accuracy"] is not None
                        else "N/A"
                    )
                ),
            "Inference Time (ms)":
                f"{result['inference_time_ms']:.2f}",
            "Model Size (MB)":
                f"{result['model_size_mb']:.2f}",
            "Grad-CAM":
                (
                    "PASS"
                    if model_name in gradcam_results
                    else "FAIL"
                ),
        }
    )

st.dataframe(
    final_metric_rows,
    width="stretch",
    hide_index=True,
)


successful_gradcam = len(
    gradcam_results
)


if (
    len(prediction_results) == 3
    and successful_gradcam == 3
):

    st.success(
        "STREAMLIT STAGE 3 PASS — "
        "Citra berhasil diproses, "
        "ketiga model berhasil melakukan "
        "prediksi, dan Grad-CAM berhasil "
        "dibuat untuk ketiga model."
    )

elif (
    len(prediction_results) == 3
):

    st.warning(
        "Prediction Engine PASS, "
        "tetapi Grad-CAM belum berhasil "
        "pada semua model."
    )

else:

    st.error(
        "STREAMLIT STAGE 3 FAIL."
    )