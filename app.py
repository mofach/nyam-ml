import os
import tempfile
import logging
from flask import Flask, request, jsonify
from io import BytesIO

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from google.cloud import storage

# =========================
# LOGGING
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
    raise RuntimeError("ENV VAR TIDAK LENGKAP")

# =========================
# FLASK INIT
# =========================
app = Flask(__name__)

# =========================
# TEMP PATH
# =========================
TMP_DIR = tempfile.gettempdir()
FOOD_MODEL_PATH = os.path.join(TMP_DIR, FOOD_MODEL_BLOB_NAME)
BMR_MODEL_PATH = os.path.join(TMP_DIR, BMR_MODEL_BLOB_NAME)

# =========================
# GLOBAL MODELS
# =========================
food_model = None
bmr_model = None

# =========================
# CLASSES
# =========================
CLASSES = [
    "ayam", "broccoli", "ikan", "kambing", "cauliflower", "potato",
    "cabbage", "pumpkin", "cucumber", "paprika", "sapi", "tofu",
    "telur", "tempeh", "tomato", "udang", "carrot"
]

# =========================
# GCS DOWNLOAD
# =========================
def download_from_gcs(bucket_name, blob_name, local_path):
    logger.info(f"Downloading gs://{bucket_name}/{blob_name}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise FileNotFoundError(f"GCS object not found: {blob_name}")

    blob.download_to_filename(local_path)
    logger.info(f"Downloaded to {local_path}")

# =========================
# STARTUP LOAD MODELS
# =========================
try:
    logger.info("STARTUP: Loading models")

    download_from_gcs(GCS_BUCKET_NAME, FOOD_MODEL_BLOB_NAME, FOOD_MODEL_PATH)
    food_model = load_model(FOOD_MODEL_PATH)
    logger.info("Food model loaded")

    download_from_gcs(GCS_BUCKET_NAME, BMR_MODEL_BLOB_NAME, BMR_MODEL_PATH)
    bmr_model = load_model(BMR_MODEL_PATH)
    logger.info("BMR model loaded")

except Exception:
    logger.exception("MODEL LOAD FAILED")
    food_model = None
    bmr_model = None

# =========================
# IMAGE PREPROCESS
# =========================
def preprocess_image(file_stream):
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "food_model_loaded": food_model is not None,
        "bmr_model_loaded": bmr_model is not None
    })

@app.route("/food", methods=["POST"])
def predict_food():
    if food_model is None:
        return jsonify({"error": "Food model not loaded"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        img = preprocess_image(BytesIO(request.files["file"].read()))
        preds = food_model.predict(img, verbose=0)[0]
        idx = int(np.argmax(preds))

        return jsonify({
            "predicted_class": CLASSES[idx],
            "confidence": float(preds[idx]),
            "all_probabilities": preds.tolist()
        })

    except Exception as e:
        logger.exception("Food prediction error")
        return jsonify({"error": str(e)}), 500

@app.route("/bmr", methods=["POST"])
def predict_bmr():
    if bmr_model is None:
        return jsonify({"error": "BMR model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    try:
        x = np.array([
            int(data["gender"]),
            float(data["height"]),
            float(data["weight"]),
            float(data["bmi"])
        ]).reshape(1, -1)

        preds = bmr_model.predict(x, verbose=0).flatten()

        return jsonify({
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
