import os
import tempfile
import logging
from flask import Flask, request, jsonify
from io import BytesIO

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from google.cloud import storage

# =========================
# LOGGING CONFIG
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# ENV VALIDATION
# =========================
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
FOOD_MODEL_BLOB_NAME = os.getenv("FOOD_MODEL_BLOB_NAME")
BMR_MODEL_BLOB_NAME = os.getenv("BMR_MODEL_BLOB_NAME")

if not all([GCS_BUCKET_NAME, FOOD_MODEL_BLOB_NAME, BMR_MODEL_BLOB_NAME]):
    raise RuntimeError(
        "ENV VAR TIDAK LENGKAP: "
        "GCS_BUCKET_NAME / FOOD_MODEL_BLOB_NAME / BMR_MODEL_BLOB_NAME"
    )

# =========================
# FLASK INIT
# =========================
app = Flask(__name__)

# =========================
# TEMP PATH (Cloud Run Safe)
# =========================
TMP_DIR = tempfile.gettempdir()
LOCAL_FOOD_MODEL_PATH = os.path.join(TMP_DIR, FOOD_MODEL_BLOB_NAME)
LOCAL_BMR_MODEL_PATH = os.path.join(TMP_DIR, BMR_MODEL_BLOB_NAME)

# =========================
# GLOBAL MODELS
# =========================
food_model = None
bmr_model = None

# =========================
# CLASSES
# =========================
classes = [
    "ayam", "broccoli", "ikan", "kambing", "cauliflower", "potato",
    "cabbage", "pumpkin", "cucumber", "paprika", "sapi", "tofu",
    "telur", "tempeh", "tomato", "udang", "carrot"
]

# =========================
# GCS DOWNLOAD
# =========================
def download_model(bucket_name, blob_name, local_path):
    logger.info(f"⬇️ Downloading gs://{bucket_name}/{blob_name}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise FileNotFoundError(f"GCS object NOT FOUND: {blob_name}")

    blob.download_to_filename(local_path)
    logger.info(f"✅ Downloaded to {local_path}")

    if not os.path.exists(local_path):
        raise RuntimeError(f"File missing after download: {local_path}")

# =========================
# STARTUP LOAD MODELS
# =========================
try:
    logger.info("🚀 STARTUP: loading ML models")

    download_model(GCS_BUCKET_NAME, FOOD_MODEL_BLOB_NAME, LOCAL_FOOD_MODEL_PATH)
    food_model = load_model(LOCAL_FOOD_MODEL_PATH)
    logger.info("🧠 Food model LOADED")

    download_model(GCS_BUCKET_NAME, BMR_MODEL_BLOB_NAME, LOCAL_BMR_MODEL_PATH)
    bmr_model = load_model(LOCAL_BMR_MODEL_PATH)
    logger.info("🧠 BMR model LOADED")

except Exception:
    logger.exception("🔥 FATAL STARTUP ERROR – MODEL LOAD FAILED")
    # Jangan raise → biar service hidup & endpoint return 503
    food_model = None
    bmr_model = None

# =========================
# IMAGE PREPROCESS
# =========================
def image_preprocess(file_stream):
    file_stream.seek(0)
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image file")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# =========================
# FOOD PREDICTION
# =========================
def predict_food(file_stream):
    img = image_preprocess(file_stream)
    img = np.expand_dims(img, axis=0)

    preds = food_model.predict(img, verbose=0)[0]
    idx = int(np.argmax(preds))

    return {
        "predicted_class": classes[idx],
        "confidence": float(preds[idx]),
        "all_probabilities": preds.tolist()
    }

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "running",
        "food_model_loaded": food_model is not None,
        "bmr_model_loaded": bmr_model is not None
    })

@app.route("/food", methods=["POST"])
def food_endpoint():
    if food_model is None:
        return jsonify({"error": "Food model not loaded"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file_stream = BytesIO(request.files["file"].read())
        result = predict_food(file_stream)

        if result["confidence"] < 0.65:
            return jsonify({"message": "Confidence < 65%"}), 400

        return jsonify(result)

    except Exception as e:
        logger.exception("Food prediction error")
        return jsonify({"error": str(e)}), 500

@app.route("/bmr", methods=["POST"])
def bmr_endpoint():
    if bmr_model is None:
        return jsonify({"error": "BMR model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    required = ["gender", "height", "weight", "bmi"]
    if not all(k in data for k in required):
        return jsonify({"error": "Missing fields"}), 400

    try:
        x = np.array([
            int(data["gender"]),
            float(data["height"]),
            float(data["weight"]),
            float(data["bmi"])
        ]).reshape(1, -1)

        preds = bmr_model.predict(x, verbose=0).flatten()

        return jsonify({
            "input": data,
            "predicted_index": int(np.argmax(preds)),
            "probabilities": preds.tolist()
        })

    except Exception as e:
        logger.exception("BMR prediction error")
        return jsonify({"error": str(e)}), 500

# =========================
# LOCAL DEV
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
