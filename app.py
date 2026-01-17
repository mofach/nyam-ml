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

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Env Vars
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
FOOD_MODEL_BLOB_NAME = os.getenv("FOOD_MODEL_BLOB_NAME")
BMR_MODEL_BLOB_NAME = os.getenv("BMR_MODEL_BLOB_NAME")

app = Flask(__name__)

# Temp Paths
temp_dir = tempfile.gettempdir()
LOCAL_FOOD_MODEL_PATH = os.path.join(temp_dir, 'modelML_foodRecognition.keras')
LOCAL_BMR_MODEL_PATH = os.path.join(temp_dir, 'modelML_bmiRate.keras')

# Globals
food_model = None
bmr_model = None
classes = [
    "ayam", "broccoli", "ikan", "kambing", "cauliflower", "potato", "cabbage", "pumpkin", 
    "cucumber", "paprika", "sapi", "tofu", "telur", "tempeh", "tomato", "udang", "carrot"
]

# --- Logic Download Cerdas (Robust) ---
def download_and_load_model(blob_name, local_path):
    logger.info(f"⬇️ Downloading {blob_name}...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            raise FileNotFoundError(f"File {blob_name} not found in bucket")
            
        blob.download_to_filename(local_path)
        logger.info(f"✅ Downloaded: {local_path}")
        
        # Load Model (TensorFlow Terbaru akan otomatis handle Keras 3)
        model = load_model(local_path)
        logger.info(f"🧠 Model Loaded: {blob_name}")
        return model
    except Exception as e:
        logger.error(f"🔥 Error loading {blob_name}: {e}")
        raise e

def get_food_model():
    global food_model
    if food_model is None:
        food_model = download_and_load_model(FOOD_MODEL_BLOB_NAME, LOCAL_FOOD_MODEL_PATH)
    return food_model

def get_bmr_model():
    global bmr_model
    if bmr_model is None:
        bmr_model = download_and_load_model(BMR_MODEL_BLOB_NAME, LOCAL_BMR_MODEL_PATH)
    return bmr_model

# --- Endpoints ---
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ML API Running (Python 3.12)", 
        "food_loaded": food_model is not None,
        "bmr_loaded": bmr_model is not None
    })

@app.route('/food', methods=['POST'])
def predict_food():
    # Load Model
    try:
        model = get_food_model()
    except Exception as e:
        return jsonify({"error": f"Model Error: {str(e)}"}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        # Preprocess (Sesuai referensi kamu)
        file_stream = BytesIO(file.read())
        file_stream.seek(0)
        file_bytes = np.frombuffer(file_stream.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None: return jsonify({'error': 'Invalid image'}), 400

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img_rgb, (224, 224))
        
        # Sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed_img = cv2.filter2D(img_resize, -1, kernel)
        
        img_batch = np.expand_dims(processed_img, axis=0)

        # Predict
        predictions = model.predict(img_batch, verbose=0)[0]
        prediction_index = np.argmax(predictions)
        confidence = float(predictions[prediction_index])
        predicted_class = classes[prediction_index]

        if confidence < 0.65:
            return jsonify({
                'error': 'Confidence too low',
                'message': 'Gambar tidak dikenali (< 65%)'
            }), 400

        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'all_probabilities': [float(p) for p in predictions]
        })
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/bmr', methods=['POST'])
def predict_bmr():
    # Load Model
    try:
        model = get_bmr_model()
    except Exception as e:
        return jsonify({"error": f"Model Error: {str(e)}"}), 503

    data = request.get_json()
    if not data: return jsonify({"error": "No data"}), 400

    try:
        gender = int(data.get('gender', 0))
        height = float(data.get('height', 0))
        weight = float(data.get('weight', 0))
        bmi = float(data.get('bmi', 0))

        input_data = np.array([gender, height, weight, bmi]).reshape(1, -1)
        
        prediction = model.predict(input_data, verbose=0).flatten()
        predicted_index = int(np.argmax(prediction))

        return jsonify({
            "input": data,
            "prediction_index": predicted_index,
            "raw_output": prediction.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)