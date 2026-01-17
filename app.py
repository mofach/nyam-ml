import os
import logging
import tempfile
from flask import Flask, request, jsonify
from io import BytesIO
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import storage
from PIL import Image

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

# --- KONFIGURASI ENV ---
# Nama Bucket & File di GCS
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
FOOD_MODEL_BLOB = os.getenv("FOOD_MODEL_BLOB_NAME", "food_model_fixed.h5")
BMR_MODEL_BLOB = os.getenv("BMR_MODEL_BLOB_NAME", "modelML_bmiRate.keras")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Lokasi simpan sementara di dalam Container (karena Container itu Read-Only kecuali /tmp)
TEMP_DIR = tempfile.gettempdir()
LOCAL_FOOD_PATH = os.path.join(TEMP_DIR, FOOD_MODEL_BLOB)
LOCAL_BMR_PATH = os.path.join(TEMP_DIR, BMR_MODEL_BLOB)

# Konfigurasi Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash') # Sesuaikan versi
else:
    logger.warning("⚠️ GEMINI_API_KEY tidak ditemukan!")

classes_list = [
    "ayam", "broccoli", "ikan", "kambing", "cauliflower", "potato", "cabbage", "pumpkin", 
    "cucumber", "paprika", "sapi", "tofu", "telur", "tempeh", "tomato", "udang", "carrot"
]
classes_str = ", ".join(classes_list)

food_model = None
bmr_model = None

# --- FUNGSI DOWNLOAD DARI GCS ---
def download_model(bucket_name, source_blob_name, destination_file_name):
    """Download model dari GCS jika belum ada di lokal"""
    if os.path.exists(destination_file_name):
        logger.info(f"📂 Model {source_blob_name} sudah ada di cache.")
        return True
        
    try:
        logger.info(f"⬇️ Downloading {source_blob_name} from GCS...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logger.info("✅ Download selesai!")
        return True
    except Exception as e:
        logger.error(f"🔥 Gagal download dari GCS: {e}")
        return False

# --- LOAD MODELS SAAT STARTUP ---
def initialize_models():
    global food_model, bmr_model
    
    # Cek apakah environment variabel GCS diset
    if not GCS_BUCKET_NAME:
        logger.error("❌ GCS_BUCKET_NAME belum diset!")
        return

    # 1. Download & Load Food Model
    if download_model(GCS_BUCKET_NAME, FOOD_MODEL_BLOB, LOCAL_FOOD_PATH):
        try:
            food_model = load_model(LOCAL_FOOD_PATH, compile=False)
            logger.info("✅ Food Model (Lokal) SIAP!")
        except Exception as e:
            logger.error(f"💀 Food Model Corrupt/Error: {e}")

    # 2. Download & Load BMR Model
    if download_model(GCS_BUCKET_NAME, BMR_MODEL_BLOB, LOCAL_BMR_PATH):
        try:
            bmr_model = load_model(LOCAL_BMR_PATH, compile=False)
            logger.info("✅ BMR Model SIAP!")
        except Exception as e:
            logger.error(f"💀 BMR Model Corrupt/Error: {e}")

# Jalankan inisialisasi
with app.app_context():
    initialize_models()

# --- LOGIC HYBRID (Sama seperti sebelumnya) ---
def predict_local(file_stream):
    try:
        file_stream.seek(0)
        file_bytes = np.frombuffer(file_stream.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img_rgb, (224, 224))
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img_processed = cv2.filter2D(img_resize, -1, kernel)
        img_batch = np.expand_dims(img_processed, axis=0)
        preds = food_model.predict(img_batch, verbose=0)[0]
        idx = np.argmax(preds)
        return classes_list[idx], float(preds[idx])
    except Exception as e:
        logger.error(f"Local Error: {e}")
        return None, 0.0

def predict_gemini(file_stream):
    try:
        file_stream.seek(0)
        img = Image.open(file_stream)
        prompt = f"Identifikasi makanan ini. Pilih satu dari: [{classes_str}]. Jika tidak ada, jawab 'unknown'."
        response = gemini_model.generate_content([prompt, img])
        result = response.text.strip().lower()
        
        if result in classes_list: return result, 0.95
        for c in classes_list:
            if c in result: return c, 0.90
        return None, 0.0
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return None, 0.0

# --- ENDPOINTS ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ready", "models_loaded": food_model is not None})

@app.route('/food', methods=['POST'])
def predict_food():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    file_stream = BytesIO(file.read())

    p_class, p_conf = None, 0.0
    
    # 1. Coba Lokal
    if food_model:
        p_class, p_conf = predict_local(file_stream)
        logger.info(f"🤖 Local: {p_class} ({p_conf:.2f})")

    # 2. Fallback Gemini
    if p_class is None or p_conf < 0.65:
        if GEMINI_API_KEY:
            logger.info("👉 Switch ke Gemini...")
            g_class, g_conf = predict_gemini(file_stream)
            if g_class:
                p_class, p_conf = g_class, g_conf
                logger.info(f"✨ Gemini: {p_class}")

    if p_class:
        return jsonify({'predicted_class': p_class, 'predicted_prob': p_conf})
    return jsonify({'message': 'Makanan tidak dikenali.'}), 400

@app.route('/bmr', methods=['POST'])
def predict_bmr():
    if not bmr_model: return jsonify({'error': 'Model loading...'}), 503
    data = request.get_json()
    try:
        input_data = np.array([int(data['gender']), float(data['height']), float(data['weight']), float(data['bmi'])]).reshape(1, -1)
        preds = bmr_model.predict(input_data, verbose=0).flatten()
        return jsonify({"prediction": {"probabilities": preds.tolist(), "predicted_index": int(np.argmax(preds))}})
    except Exception as e: return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)