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
from dotenv import load_dotenv

# Setup Logging biar error kelihatan di Cloud Run Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env (Hanya ngefek di local, di Cloud Run dilewati)
load_dotenv()

# --- HAPUS BAGIAN OS.ENVIRON CREDENTIALS ---
# Cloud Run otomatis handle auth, tidak perlu set manual!

# Konfigurasi Env
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
FOOD_MODEL_BLOB_NAME = os.getenv("FOOD_MODEL_BLOB_NAME")
BMR_MODEL_BLOB_NAME = os.getenv("BMR_MODEL_BLOB_NAME")

# Inisialisasi Flask
app = Flask(__name__)

# Direktori Sementara
temp_dir = tempfile.gettempdir()
LOCAL_FOOD_MODEL_PATH = os.path.join(temp_dir, 'food_model.keras')
LOCAL_BMR_MODEL_PATH = os.path.join(temp_dir, 'bmr_model.keras')

# Variable Global
food_model = None
bmr_model = None
classes = [
    "ayam", "broccoli", "ikan", "kambing", "cauliflower", "potato", "cabbage", "pumpkin", 
    "cucumber", "paprika", "sapi", "tofu", "telur", "tempeh", "tomato", "udang", "carrot"
]

def download_and_load(bucket_name, blob_name, local_path):
    """Fungsi download yang aman dengan logging jelas"""
    try:
        logger.info(f"⬇️ Downloading {blob_name} from {bucket_name}...")
        storage_client = storage.Client() # Otomatis pakai Auth Cloud Run
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            logger.error(f"❌ File {blob_name} TIDAK ADA di bucket {bucket_name}")
            return None

        blob.download_to_filename(local_path)
        logger.info(f"✅ Downloaded to: {local_path}")
        
        # Load Model
        model = load_model(local_path)
        logger.info(f"🧠 Model Loaded: {blob_name}")
        return model
    except Exception as e:
        logger.error(f"🔥 Error loading {blob_name}: {e}")
        return None

# --- Load Model saat Startup ---
logger.info("⏳ Starting Model Download...")

# Cek apakah Env Var terbaca
if not GCS_BUCKET_NAME:
    logger.critical("⚠️ GCS_BUCKET_NAME is not set! Check deploy.yml")

# Download Food Model
food_model = download_and_load(GCS_BUCKET_NAME, FOOD_MODEL_BLOB_NAME, LOCAL_FOOD_MODEL_PATH)

# Download BMR Model
bmr_model = download_and_load(GCS_BUCKET_NAME, BMR_MODEL_BLOB_NAME, LOCAL_BMR_MODEL_PATH)

if food_model and bmr_model:
    logger.info("🚀 SYSTEM READY: All models loaded.")
else:
    logger.warning("⚠️ SYSTEM PARTIAL: Some models failed to load.")

# --- Helper Functions ---
def preprocess_image(file_stream):
    file_stream.seek(0)
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_rgb, (224, 224))
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img_resize, -1, kernel)

# --- Endpoints ---
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ML API Running (Fixed Credentials)",
        "bucket": GCS_BUCKET_NAME,
        "food_model_ready": food_model is not None,
        "bmr_model_ready": bmr_model is not None
    })

@app.route('/food', methods=['POST'])
def predict_food():
    if food_model is None:
        return jsonify({'error': 'Food Model not ready. Check server logs.'}), 503

    if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400

    try:
        file = request.files['file']
        processed_img = preprocess_image(BytesIO(file.read()))
        img_batch = np.expand_dims(processed_img, axis=0)

        predictions = food_model.predict(img_batch, verbose=0)[0]
        idx = np.argmax(predictions)
        confidence = float(predictions[idx])
        
        if confidence < 0.65:
            return jsonify({'message': 'Confidence too low (<65%)'}), 400

        return jsonify({
            'class': classes[idx],
            'confidence': confidence,
            'probabilities': [float(p) for p in predictions]
        })
    except Exception as e:
        logger.error(f"Predict Food Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/bmr', methods=['POST'])
def predict_bmr():
    if bmr_model is None:
        return jsonify({'error': 'BMR Model not ready. Check server logs.'}), 503

    data = request.get_json()
    try:
        input_data = np.array([
            int(data['gender']), float(data['height']), 
            float(data['weight']), float(data['bmi'])
        ]).reshape(1, -1)
        
        preds = bmr_model.predict(input_data, verbose=0).flatten()
        return jsonify({
            "prediction_index": int(np.argmax(preds)),
            "probabilities": preds.tolist()
        })
    except Exception as e:
        logger.error(f"Predict BMR Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)