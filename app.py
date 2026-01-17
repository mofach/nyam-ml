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

# Direktori Sementara (Sesuai kode kamu)
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

# --- FUNGSI BARU: Download & Load Cerdas ---
def download_and_load_model(blob_name, local_path):
    """Mencoba download model dari GCS dan load ke Memory"""
    logger.info(f"⬇️ Mencoba download {blob_name} dari bucket {GCS_BUCKET_NAME}...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        
        # Cek apakah file ada di GCS
        if not blob.exists():
            raise FileNotFoundError(f"❌ File {blob_name} TIDAK DITEMUKAN di bucket {GCS_BUCKET_NAME}")
            
        blob.download_to_filename(local_path)
        logger.info(f"✅ Download sukses: {local_path}")
        
        model = load_model(local_path)
        logger.info(f"🧠 Model {blob_name} berhasil dimuat ke Memory!")
        return model
    except Exception as e:
        logger.error(f"🔥 Gagal load model {blob_name}: {str(e)}")
        # Lempar error agar bisa ditangkap endpoint
        raise e

def get_food_model():
    """Getter Food Model (Lazy Load)"""
    global food_model
    if food_model is None:
        food_model = download_and_load_model(FOOD_MODEL_BLOB_NAME, LOCAL_FOOD_MODEL_PATH)
    return food_model

def get_bmr_model():
    """Getter BMR Model (Lazy Load)"""
    global bmr_model
    if bmr_model is None:
        bmr_model = download_and_load_model(BMR_MODEL_BLOB_NAME, LOCAL_BMR_MODEL_PATH)
    return bmr_model

# --- Endpoints ---
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ML API is running",
        "bucket": GCS_BUCKET_NAME,
        "food_model_loaded": food_model is not None,
        "bmr_model_loaded": bmr_model is not None
    }), 200

@app.route('/food', methods=['POST'])
def predict_food():
    # 1. Pastikan Model Siap
    try:
        model = get_food_model()
    except Exception as e:
        return jsonify({"error": f"Gagal memuat Model Makanan. Cek Logs/Izin GCS. Detail: {str(e)}"}), 503

    # 2. Cek File
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    try:
        # 3. Preprocess (Logic kamu)
        file_stream = BytesIO(file.read())
        file_stream.seek(0)
        file_bytes = np.frombuffer(file_stream.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None: return jsonify({'error': 'File bukan gambar valid'}), 400

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img_rgb, (224, 224))
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed_img = cv2.filter2D(img_resize, -1, kernel)
        
        img_batch = np.expand_dims(processed_img, axis=0)

        # 4. Predict
        predictions = model.predict(img_batch, verbose=0)[0]
        prediction_index = np.argmax(predictions)
        confidence = float(predictions[prediction_index])
        predicted_class = classes[prediction_index]

        if confidence < 0.65:
            return jsonify({
                'error': 'Confidence too low',
                'message': 'Gambar tidak dikenali atau keyakinan < 65%.'
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
    # 1. Pastikan Model Siap
    try:
        model = get_bmr_model()
    except Exception as e:
        return jsonify({"error": f"Gagal memuat Model BMR. Cek Logs/Izin GCS. Detail: {str(e)}"}), 503

    data = request.get_json()
    required = ['gender', 'height', 'weight', 'bmi']
    
    if not data or not all(k in data for k in required):
        return jsonify({"error": f"Input harus lengkap: {required}"}), 400

    try:
        gender = int(data['gender'])
        height = float(data['height'])
        weight = float(data['weight'])
        bmi = float(data['bmi'])

        input_data = np.array([gender, height, weight, bmi]).reshape(1, -1)
        
        prediction = model.predict(input_data, verbose=0).flatten()
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
    app.run(host='0.0.0.0', port=8080)