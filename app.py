import os
import tempfile
import logging
from flask import Flask, request, jsonify
from io import BytesIO

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
FOOD_MODEL_BLOB_NAME = os.getenv("FOOD_MODEL_BLOB_NAME")
BMR_MODEL_BLOB_NAME = os.getenv("BMR_MODEL_BLOB_NAME")

TMP_DIR = tempfile.gettempdir()
FOOD_MODEL_PATH = os.path.join(TMP_DIR, FOOD_MODEL_BLOB_NAME or "food.keras")
BMR_MODEL_PATH = os.path.join(TMP_DIR, BMR_MODEL_BLOB_NAME or "bmr.keras")

food_model = None
bmr_model = None

CLASSES = [
    "ayam", "broccoli", "ikan", "kambing", "cauliflower", "potato",
    "cabbage", "pumpkin", "cucumber", "paprika", "sapi", "tofu",
    "telur", "tempeh", "tomato", "udang", "carrot"
]

def download_from_gcs(blob_name, local_path):
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise FileNotFoundError(blob_name)

    blob.download_to_filename(local_path)

def load_food_model():
    global food_model
    if food_model is None:
        logger.info("Loading FOOD model (lazy)")
        download_from_gcs(FOOD_MODEL_BLOB_NAME, FOOD_MODEL_PATH)
        food_model = load_model(FOOD_MODEL_PATH)

def load_bmr_model():
    global bmr_model
    if bmr_model is None:
        logger.info("Loading BMR model (lazy)")
        download_from_gcs(BMR_MODEL_BLOB_NAME, BMR_MODEL_PATH)
        bmr_model = load_model(BMR_MODEL_PATH)

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "food_model_loaded": food_model is not None,
        "bmr_model_loaded": bmr_model is not None
    })

@app.route("/bmr", methods=["POST"])
def predict_bmr():
    try:
        load_bmr_model()
    except Exception as e:
        logger.exception("BMR model load failed")
        return jsonify({"error": "Model load failed", "detail": str(e)}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

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

@app.route("/food", methods=["POST"])
def predict_food():
    try:
        load_food_model()
    except Exception as e:
        logger.exception("Food model load failed")
        return jsonify({"error": "Model load failed", "detail": str(e)}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    img = cv2.imdecode(
        np.frombuffer(request.files["file"].read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    preds = food_model.predict(img, verbose=0)[0]
    idx = int(np.argmax(preds))

    return jsonify({
        "predicted_class": CLASSES[idx],
        "confidence": float(preds[idx]),
        "all_probabilities": preds.tolist()
    })
