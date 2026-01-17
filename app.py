import os
import tempfile
import logging
from flask import Flask, request, jsonify
from io import BytesIO
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from google.cloud import storage
from dotenv import load_dotenv

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Konfigurasi Env
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
FOOD_MODEL_BLOB_NAME = os.getenv("FOOD_MODEL_BLOB_NAME")
BMR_MODEL_BLOB_NAME = os.getenv("BMR_MODEL_BLOB_NAME")

# Inisialisasi Flask
app = Flask(__name__)

# Direktori Sementara
temp_dir = tempfile.gettempdir()
LOCAL_FOOD_MODEL_PATH = os.path.join(temp_dir, 'modelML_foodRecognition.keras')
LOCAL_BMR_MODEL_PATH = os.path.join(temp_dir, 'modelML_bmiRate.keras')

# Variable Global Model
food_model = None
bmr_model = None
classes = [
    "ayam", "broccoli", "ikan", "kambing", "cauliflower", "potato", "cabbage", "pumpkin", 
    "cucumber", "paprika", "sapi", "tofu", "telur", "tempeh", "tomato", "udang", "carrot"
]

def download_model(bucket_name, source_blob_name, destination_file_name):
    """Helper: Download file dari GCS"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logger.info(f"✅ Model {source_blob_name} berhasil didownload ke {destination_file_name}")
    except Exception as e:
        logger.error(f"❌ Gagal download model {source_blob_name}: {e}")
        raise e

def initialize_models():
    """Load model ke memori (Global)"""
    global food_model, bmr_model
    
    # Download jika belum ada di temp (berguna buat local testing, di cloud run temp selalu kosong pas start)
    if not os.path.exists(LOCAL_FOOD_MODEL_PATH):
        download_model(GCS_BUCKET_NAME, FOOD_MODEL_BLOB_NAME, LOCAL_FOOD_MODEL_PATH)
    
    if not os.path.exists(LOCAL_BMR_MODEL_PATH):
        download_model(GCS_BUCKET_NAME, BMR_MODEL_BLOB_NAME, LOCAL_BMR_MODEL_PATH)

    logger.info("⏳ Memuat model ke TensorFlow...")
    food_model = load_model(LOCAL_FOOD_MODEL_PATH)
    bmr_model = load_model(LOCAL_BMR_MODEL_PATH)
    logger.info("🚀 Semua model siap digunakan!")

# --- Load Model saat Startup ---
# Di Cloud Run, ini akan dijalankan saat container start (Cold Start)
try:
    initialize_models()
except Exception as e:
    logger.critical(f"FATAL: Gagal inisialisasi model. {e}")

# --- Helper Functions ---
def preprocess_image(file_stream):
    file_stream.seek(0)
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("File bukan gambar yang valid")

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_rgb, (224, 224))
    
    # Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(img_resize, -1, kernel)
    
    return sharpened_image

# --- Endpoints ---
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "ML API is running", "models_loaded": food_model is not None}), 200

@app.route('/food', methods=['POST'])
def predict_food():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    try:
        # Preprocess
        processed_img = preprocess_image(BytesIO(file.read()))
        img_batch = np.expand_dims(processed_img, axis=0)

        # Predict
        predictions = food_model.predict(img_batch, verbose=0)[0]
        prediction_index = np.argmax(predictions)
        confidence = float(predictions[prediction_index])
        predicted_class = classes[prediction_index]

        if confidence < 0.65:
            return jsonify({
                'error': 'Confidence too low',
                'message': 'Gambar tidak dikenali atau keyakinan < 65%. Coba ambil gambar lebih jelas.'
            }), 400

        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'all_probabilities': [float(p) for p in predictions]
        })

    except Exception as e:
        logger.error(f"Error predict food: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/bmr', methods=['POST'])
def predict_bmr_endpoint():
    data = request.get_json()
    required = ['gender', 'height', 'weight', 'bmi']
    
    if not data or not all(k in data for k in required):
        return jsonify({"error": f"Input harus lengkap: {required}"}), 400

    try:
        # PENTING: Gunakan Float untuk BMI agar presisi
        gender = int(data['gender'])
        height = float(data['height']) # Bisa koma
        weight = float(data['weight']) # Bisa koma
        bmi = float(data['bmi'])       # Wajib Float

        input_data = np.array([gender, height, weight, bmi]).reshape(1, -1)
        
        # Predict BMR
        prediction = bmr_model.predict(input_data, verbose=0).flatten()
        # Asumsi output model BMR kamu adalah klasifikasi (softmax) karena pakai argmax
        predicted_index = int(np.argmax(prediction))

        return jsonify({
            "input": data,
            "prediction_index": predicted_index,
            "raw_output": prediction.tolist()
        })

    except Exception as e:
        logger.error(f"Error predict BMR: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ini hanya untuk testing lokal, production pakai Gunicorn
    app.run(host='0.0.0.0', port=8080, debug=True)