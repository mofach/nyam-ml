import os
import tempfile
import logging
from flask import Flask, request, jsonify
from io import BytesIO
import numpy as np
import cv2

# SUPPRESS TensorFlow warnings SEBELUM import TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress semua warning TF
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import load_model
from google.cloud import storage
from dotenv import load_dotenv

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Konfigurasi
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
FOOD_MODEL_BLOB_NAME = os.getenv("FOOD_MODEL_BLOB_NAME")
BMR_MODEL_BLOB_NAME = os.getenv("BMR_MODEL_BLOB_NAME")

app = Flask(__name__)

# Direktori Sementara
temp_dir = tempfile.gettempdir()
LOCAL_FOOD_MODEL_PATH = os.path.join(temp_dir, FOOD_MODEL_BLOB_NAME)
LOCAL_BMR_MODEL_PATH = os.path.join(temp_dir, BMR_MODEL_BLOB_NAME)

# Variable Global
food_model = None
bmr_model = None
models_loading = True  # Flag untuk track loading status

classes = [
    "ayam", "broccoli", "ikan", "kambing", "cauliflower", "potato", "cabbage", "pumpkin", 
    "cucumber", "paprika", "sapi", "tofu", "telur", "tempeh", "tomato", "udang", "carrot"
]

# ===========================
# Fungsi Download dengan Retry
# ===========================
def download_model_from_gcs(bucket_name, source_blob_name, destination_file_name, max_retries=3):
    """Download model dari GCS dengan retry mechanism"""
    if not bucket_name or not source_blob_name:
        logger.error("❌ Bucket name atau blob name tidak tersedia")
        return False
    
    # Cek apakah file sudah ada
    if os.path.exists(destination_file_name):
        file_size = os.path.getsize(destination_file_name) / (1024 * 1024)  # MB
        logger.info(f"✅ Model sudah ada di {destination_file_name} ({file_size:.2f} MB)")
        return True
    
    for attempt in range(max_retries):
        try:
            logger.info(f"⬇️ Downloading {source_blob_name} (attempt {attempt + 1}/{max_retries})...")
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(source_blob_name)
            
            # Download
            blob.download_to_filename(destination_file_name)
            
            # Verifikasi file berhasil di-download
            if os.path.exists(destination_file_name) and os.path.getsize(destination_file_name) > 0:
                file_size = os.path.getsize(destination_file_name) / (1024 * 1024)  # MB
                logger.info(f"✅ Downloaded {source_blob_name} ({file_size:.2f} MB)")
                return True
            else:
                logger.error(f"❌ File downloaded tapi kosong atau tidak ada")
                return False
                
        except Exception as e:
            logger.error(f"🔥 Error downloading {source_blob_name} (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return False
    
    return False

# ===========================
# Load Models dengan Error Handling
# ===========================
def load_models():
    """Load models dengan proper error handling"""
    global food_model, bmr_model, models_loading
    
    try:
        logger.info("🚀 Starting model initialization...")
        logger.info(f"📦 Bucket: {GCS_BUCKET_NAME}")
        logger.info(f"📦 Food Model: {FOOD_MODEL_BLOB_NAME}")
        logger.info(f"📦 BMR Model: {BMR_MODEL_BLOB_NAME}")
        
        # Download Food Model
        if download_model_from_gcs(GCS_BUCKET_NAME, FOOD_MODEL_BLOB_NAME, LOCAL_FOOD_MODEL_PATH):
            try:
                logger.info("🔄 Loading food model...")
                food_model = load_model(LOCAL_FOOD_MODEL_PATH, compile=False)
                logger.info("✅ Food Model Loaded Successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load food model: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.error("❌ Failed to download food model")
        
        # Download BMR Model
        if download_model_from_gcs(GCS_BUCKET_NAME, BMR_MODEL_BLOB_NAME, LOCAL_BMR_MODEL_PATH):
            try:
                logger.info("🔄 Loading BMR model...")
                bmr_model = load_model(LOCAL_BMR_MODEL_PATH, compile=False)
                logger.info("✅ BMR Model Loaded Successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load BMR model: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.error("❌ Failed to download BMR model")
        
        # Status check
        models_loading = False
        if food_model is None and bmr_model is None:
            logger.critical("🔥 CRITICAL: No models loaded successfully!")
        else:
            logger.info(f"✅ Startup Complete - Food: {food_model is not None}, BMR: {bmr_model is not None}")
            
    except Exception as e:
        models_loading = False
        logger.critical(f"🔥 FATAL STARTUP ERROR: {str(e)}")
        import traceback
        logger.critical(traceback.format_exc())

# Load models at startup (async-like)
import threading
model_thread = threading.Thread(target=load_models, daemon=True)
model_thread.start()

# ===========================
# Helper Functions
# ===========================
def ImagePreprocess1(file_stream):
    file_stream.seek(0)
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Invalid image file")
    
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
        logger.error(f"Prediction error: {e}")
        return None, None, None, str(e)

# ===========================
# Endpoint API
# ===========================
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "Running",
        "python_version": "3.12",
        "models_loading": models_loading,
        "models_ready": {
            "food_model": food_model is not None,
            "bmr_model": bmr_model is not None
        },
        "config": {
            "bucket": GCS_BUCKET_NAME,
            "food_model_file": FOOD_MODEL_BLOB_NAME,
            "bmr_model_file": BMR_MODEL_BLOB_NAME
        }
    }), 200

@app.route('/food', methods=['POST'])
def predict():
    """Food prediction endpoint"""
    
    # Check if models are still loading
    if models_loading:
        return jsonify({
            'error': 'Models are still loading. Please wait a moment and try again.',
            'status': 'loading'
        }), 503
    
    # Check if food model is loaded
    if food_model is None:
        logger.error("Food model not loaded")
        return jsonify({
            'error': 'Food Model failed to load. Check server logs for details.',
            'status': 'unavailable'
        }), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        file_stream = BytesIO(file.read())
        probs, p_class, p_prob, error = predict_image(file_stream)
        
        if error:
            return jsonify({'error': error}), 500
        
        if p_class:
            if p_prob < 0.65:
                return jsonify({
                    'message': 'Prediksi kurang dari 65% kepercayaan.',
                    'predicted_class': p_class,
                    'confidence': p_prob
                }), 400
            
            return jsonify({
                'all_probabilities': probs,
                'predicted_class': p_class,
                'predicted_prob': p_prob
            }), 200
        
        return jsonify({'error': 'Prediction failed'}), 500
        
    except Exception as e:
        logger.error(f"Request error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/bmr', methods=['POST'])
def predict_bmr():
    """BMR prediction endpoint"""
    
    # Check if models are still loading
    if models_loading:
        return jsonify({
            'error': 'Models are still loading. Please wait a moment and try again.',
            'status': 'loading'
        }), 503
    
    # Check if BMR model is loaded
    if bmr_model is None:
        logger.error("BMR model not loaded")
        return jsonify({
            'error': 'BMR Model failed to load. Check server logs for details.',
            'status': 'unavailable'
        }), 503

    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    if not all(key in data for key in ['gender', 'height', 'weight', 'bmi']):
        return jsonify({
            "error": "Data input tidak lengkap. Required: gender, height, weight, bmi",
            "received": list(data.keys()) if data else []
        }), 400

    try:
        input_data = np.array([
            int(data['gender']), 
            float(data['height']), 
            float(data['weight']), 
            float(data['bmi'])
        ]).reshape(1, -1)
        
        logger.info(f"BMR prediction request: {data}")
        preds = bmr_model.predict(input_data, verbose=0).flatten()
        
        return jsonify({
            "input": data,
            "prediction": {
                "probabilities": preds.tolist(),
                "predicted_index": int(np.argmax(preds))
            }
        }), 200
        
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"BMR prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)