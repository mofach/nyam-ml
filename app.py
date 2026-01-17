import os
import tempfile
from flask import Flask, request, jsonify
from io import BytesIO
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from google.cloud import storage
from dotenv import load_dotenv
from os import environ

# Load variabel dari file .env
load_dotenv()

# Konfigurasi dari .env
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
FOOD_MODEL_BLOB_NAME = os.getenv("FOOD_MODEL_BLOB_NAME")
BMR_MODEL_BLOB_NAME = os.getenv("BMR_MODEL_BLOB_NAME")

# Tentukan direktori sementara
temp_dir = tempfile.gettempdir()
# Pastikan nama file environment variable kamu SUDAH termasuk ekstensi .keras/.h5
LOCAL_FOOD_MODEL_PATH = os.path.join(temp_dir, FOOD_MODEL_BLOB_NAME)
LOCAL_BMR_MODEL_PATH = os.path.join(temp_dir, BMR_MODEL_BLOB_NAME)

# Inisialisasi Flask
app = Flask(__name__)

# ===========================
# Fungsi untuk Unduh Model
# ===========================
def download_model_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Unduh file dari Google Cloud Storage ke lokal."""
    # Pastikan direktori tujuan ada
    destination_dir = os.path.dirname(destination_file_name)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Unduh file dari GCS
    print(f"⬇️ Downloading {source_blob_name}...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"✅ Model {source_blob_name} downloaded to {destination_file_name}.")

# Initialize models as None
food_model = None
bmr_model = None

# Load models at startup (Sesuai request kamu)
try:
    download_model_from_gcs(GCS_BUCKET_NAME, FOOD_MODEL_BLOB_NAME, LOCAL_FOOD_MODEL_PATH)
    download_model_from_gcs(GCS_BUCKET_NAME, BMR_MODEL_BLOB_NAME, LOCAL_BMR_MODEL_PATH)
    
    # Load ke Memory
    food_model = load_model(LOCAL_FOOD_MODEL_PATH)
    bmr_model = load_model(LOCAL_BMR_MODEL_PATH)
    print("🚀 Models loaded successfully into memory")
except Exception as e:
    print(f"🔥 Error during startup: {str(e)}")
    # Warning: Kalau gagal di sini, server mungkin akan crash atau variabel None

# Daftar kelas
classes = [
    "ayam", "broccoli", "ikan", "kambing", "cauliflower", "potato", "cabbage", "pumpkin", 
    "cucumber", "paprika", "sapi", "tofu", "telur", "tempeh", "tomato", "udang", "carrot"
]

# ===========================
# Fungsi Pendukung
# ===========================

def ImagePreprocess1(file_stream):
    file_stream.seek(0)
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_rgb, (224, 224))
    
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(img_resize, -1, kernel)
    
    return sharpened_image

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
        "status": "Running (TF 2.15)",
        "models_ready": food_model is not None and bmr_model is not None
    })

@app.route('/food', methods=['POST'])
def predict():
    if food_model is None:
        return jsonify({'error': 'Model belum siap/gagal download. Cek logs.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        file_stream = BytesIO(file.read())
        all_probabilities, predicted_class, predicted_prob, error = predict_image(file_stream)
        
        if error:
            return jsonify({'error': error}), 500
            
        if predicted_class:
            if predicted_prob < 0.65:
                return jsonify({
                    'message': 'Prediksi kurang dari 65% kepercayaan. Silakan kirim ulang gambar.'
                }), 400
            
            return jsonify({
                'all_probabilities': all_probabilities,
                'predicted_class': predicted_class,
                'predicted_prob': predicted_prob
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/bmr', methods=['POST'])
def predict_bmr():
    if bmr_model is None:
        return jsonify({'error': 'Model BMR belum siap.'}), 503

    data = request.get_json()
    if not all(key in data for key in ['gender', 'height', 'weight', 'bmi']):
        return jsonify({"error": "Data input tidak lengkap"}), 400

    try:
        gender = int(data['gender'])  
        height = float(data['height']) # Pakai float biar aman 
        weight = float(data['weight'])  
        bmi = float(data['bmi'])

        input_data = np.array([gender, height, weight, bmi]).reshape(1, -1)
        prediction = bmr_model.predict(input_data, verbose=0).flatten()
        predicted_index = int(np.argmax(prediction))

        return jsonify({
            "input": data,
            "prediction": {
                "probabilities": prediction.tolist(),
                "predicted_index": predicted_index
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)