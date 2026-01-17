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

# Set kredensial Google Cloud Storage
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Konfigurasi dari .env
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
FOOD_MODEL_BLOB_NAME = os.getenv("FOOD_MODEL_BLOB_NAME")
BMR_MODEL_BLOB_NAME = os.getenv("BMR_MODEL_BLOB_NAME")

# Tentukan direktori sementara
temp_dir = tempfile.gettempdir()
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
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Model {source_blob_name} downloaded to {destination_file_name}.")

# Initialize models as None (lazy loading)
food_model = None
bmr_model = None

# Load models at startup
try:
    download_model_from_gcs(GCS_BUCKET_NAME, FOOD_MODEL_BLOB_NAME, LOCAL_FOOD_MODEL_PATH)
    download_model_from_gcs(GCS_BUCKET_NAME, BMR_MODEL_BLOB_NAME, LOCAL_BMR_MODEL_PATH)
    print("Models downloaded successfully")
except Exception as e:
    print(f"Error downloading models: {str(e)}")

# Daftar kelas sesuai output model
classes = [
    "ayam", "broccoli", "ikan", "kambing", "cauliflower", "potato", "cabbage", "pumpkin", 
    "cucumber", "paprika", "sapi", "tofu", "telur", "tempeh", "tomato", "udang", "carrot"
]

# ===========================
# Fungsi Pendukung
# ===========================

def load_models():
    """Load models if they haven't been loaded yet."""
    global food_model, bmr_model
    if food_model is None or bmr_model is None:
        # Download and load models
        download_model_from_gcs(GCS_BUCKET_NAME, FOOD_MODEL_BLOB_NAME, LOCAL_FOOD_MODEL_PATH)
        download_model_from_gcs(GCS_BUCKET_NAME, BMR_MODEL_BLOB_NAME, LOCAL_BMR_MODEL_PATH)
        
        food_model = load_model(LOCAL_FOOD_MODEL_PATH)
        bmr_model = load_model(LOCAL_BMR_MODEL_PATH)

# Initialize models at startup
with app.app_context():
    load_models()

# Fungsi preprocessing gambar
def ImagePreprocess1(file_stream):
    # Pastikan pointer stream berada di awal
    file_stream.seek(0)
    # Membaca file gambar dari stream
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Konversi BGR ke RGB dan ubah ukurannya
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_rgb, (224, 224))
    
    # Filter sharpening untuk mempertajam gambar
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(img_resize, -1, kernel)
    
    return sharpened_image

# Fungsi prediksi gambar
def predict_image(file_stream):
    try:
        # Preprocessing gambar
        processed_image = ImagePreprocess1(file_stream)
        processed_image = np.expand_dims(processed_image, axis=0)  # Tambahkan dimensi batch
        
        # Prediksi menggunakan model
        predictions = food_model.predict(processed_image, verbose=0)[0]
        
        # Mendapatkan kelas dengan nilai tertinggi
        prediction_index = np.argmax(predictions)
        predicted_class = classes[prediction_index]
        confidence = float(predictions[prediction_index])  # Konversi ke float

        # Konversi probabilitas ke list float
        all_probabilities = [float(prob) for prob in predictions]
        
        return all_probabilities, predicted_class, confidence, None
    except Exception as e:
        return None, None, None, str(e)

# ===========================
# Endpoint API
# ===========================
@app.route('/food', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        # Proses prediksi
        file_stream = BytesIO(file.read())
        all_probabilities, predicted_class, predicted_prob, error = predict_image(file_stream)
        if predicted_class:
            # Cek jika predicted_prob kurang dari 65%
            if predicted_prob < 0.65:
                return jsonify({
                    'message': 'Prediksi kurang dari 65% kepercayaan. Silakan kirim ulang gambar.'
                }), 400  # Mengembalikan status 400 Bad Request
            return jsonify({
                'all_probabilities': all_probabilities,
                'predicted_class': predicted_class,
                'predicted_prob': predicted_prob
            })
        else:
            return jsonify({'error': error}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/bmr', methods=['POST'])
def predict_bmr():
    """Endpoint untuk prediksi BMR."""
    data = request.get_json()

    # Validasi input
    if not all(key in data for key in ['gender', 'height', 'weight', 'bmi']):
        return jsonify({"error": "Data input harus memiliki 'gender', 'height', 'weight' dan 'BMI'"}), 400

    gender = int(data['gender'])  
    height = int(data['height'])  
    weight = int(data['weight'])  
    bmi = int(data['bmi'])

    input_data = np.array([gender, height, weight, bmi]).reshape(1, -1)
    prediction = bmr_model.predict(input_data).flatten()
    predicted_index = int(np.argmax(prediction))

    return jsonify({
        "input": {
            "gender": gender,
            "height": height,
            "weight": weight,
            "bmi": bmi
        },
        "prediction": {
            "probabilities": prediction.tolist(),
            "predicted_index": predicted_index
        }
    })

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8080)