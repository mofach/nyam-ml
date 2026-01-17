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

# Setup Logging (Biar kelihatan di Cloud Run)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env (Cuma ngefek di laptop, server bakal skip ini)
load_dotenv()

# --- BAGIAN INI SAYA HAPUS BIAR GAK CRASH DI SERVER ---
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ... (HAPUS)

# Konfigurasi Env
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
FOOD_MODEL_BLOB_NAME = os.getenv("FOOD_MODEL_BLOB_NAME")
BMR_MODEL_BLOB_NAME = os.getenv("BMR_MODEL_BLOB_NAME")

# Inisialisasi Flask
app = Flask(__name__)

# Direktori Sementara
temp_dir = tempfile.gettempdir()
LOCAL_FOOD_MODEL_PATH = os.path.join(temp_dir, FOOD_MODEL_BLOB_NAME)
LOCAL_BMR_MODEL_PATH = os.path.join(temp_dir, BMR_MODEL_BLOB_NAME)

# Variable Global
food_model = None
bmr_model = None

classes = [
    "ayam", "broccoli", "ikan", "kambing", "cauliflower", "potato", "cabbage", "pumpkin", 
    "cucumber", "paprika", "sapi", "tofu", "telur", "tempeh", "tomato", "udang", "carrot"
]

# ===========================
# Fungsi Download
# ===========================
def download_model_from_gcs(bucket_name, source_blob_name, destination_file_name):
    try:
        logger.info(f"⬇️ Downloading {source_blob_name}...")
        # Di Cloud Run otomatis pakai Auth Server, gak perlu creds manual
        storage_client = storage.Client() 
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logger.info(f"✅ Downloaded to {destination_file_name}")
        return True
    except Exception as e:
        logger.error(f"🔥 Error downloading {source_blob_name}: {e}")
        return False

# Load models at startup
try:
    dl_food = download_model_from_gcs(GCS_BUCKET_NAME, FOOD_MODEL_BLOB_NAME, LOCAL_FOOD_MODEL_PATH)
    dl_bmr = download_model_from_gcs(GCS_BUCKET_NAME, BMR_MODEL_BLOB_NAME, LOCAL_BMR_MODEL_PATH)
    
    if dl_food:
        food_model = load_model(LOCAL_FOOD_MODEL_PATH)
        logger.info("🧠 Food Model Loaded")
        
    if dl_bmr:
        bmr_model = load_model(LOCAL_BMR_MODEL_PATH)
        logger.info("🧠 BMR Model Loaded")
        
except Exception as e:
    logger.critical(f"🔥 FATAL STARTUP ERROR: {str(e)}")

# ===========================
# Helper Functions (Sama Persis Punya Kamu)
# ===========================
def ImagePreprocess1(file_stream):
    file_stream.seek(0)
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_rgb, (224, 224))
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img_resize, -1, kernel)

def predict_image(file_stream):
    try:
        processed_image = ImagePreprocess1(file_stream)
        processed_image = np.expand_dims(processed_image, axis=0)
        predictions = food_model.predict(processed_image, verbose=0)[0]
        prediction_index = np.argmax(predictions)
        predicted_class = classes[prediction_index]
        confidence = float(predictions[prediction_index])
        all_probabilities = [float(prob) for prob in predictions]
        return all_probabilities, predicted_class, confidence, None
    except Exception as e:
        return None, None, None, str(e)

# ===========================
# Endpoint API
# ===========================
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "Running (Python 3.12)",
        "models_ready": food_model is not None and bmr_model is not None
    })

@app.route('/food', methods=['POST'])
def predict():
    if food_model is None: return jsonify({'error': 'Food Model Failed to Load'}), 503
    if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        file_stream = BytesIO(file.read())
        probs, p_class, p_prob, error = predict_image(file_stream)
        
        if error: return jsonify({'error': error}), 500
        if p_class:
            if p_prob < 0.65:
                return jsonify({'message': 'Prediksi kurang dari 65% kepercayaan.'}), 400
            return jsonify({
                'all_probabilities': probs,
                'predicted_class': p_class,
                'predicted_prob': p_prob
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/bmr', methods=['POST'])
def predict_bmr():
    if bmr_model is None: return jsonify({'error': 'BMR Model Failed to Load'}), 503

    data = request.get_json()
    if not all(key in data for key in ['gender', 'height', 'weight', 'bmi']):
        return jsonify({"error": "Data input tidak lengkap"}), 400

    try:
        input_data = np.array([int(data['gender']), float(data['height']), float(data['weight']), float(data['bmi'])]).reshape(1, -1)
        preds = bmr_model.predict(input_data, verbose=0).flatten()
        return jsonify({
            "input": data,
            "prediction": {
                "probabilities": preds.tolist(),
                "predicted_index": int(np.argmax(preds))
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)