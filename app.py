from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

app = Flask(__name__)

# Load the model and class names
model = load_model("lung_cancer_resnet.h5", compile=False)
with open("labels.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the uploads directory exists

# Preprocessing function
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    size = (150, 150)  # Model input size
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)
    normalized_image_array = (image_array / 127.5) - 1  # Normalize to [-1, 1]
    data = np.expand_dims(normalized_image_array, axis=0)  # Shape: (1, 150, 150, 3)
    return data

uploaded_files = []

@app.route('/predict', methods=['POST'])
def predict():
    global uploaded_files
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        if not files:
            return jsonify({"error": "No files uploaded."}), 400

        predictions = []
        for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            uploaded_files.append(file_path)

            # Preprocess the image and make predictions
            data = preprocess_image(file_path)
            prediction = model.predict(data)

            # Get the predicted class index and name
            index = np.argmax(prediction)
            if 0 <= index < len(class_names):
                class_name = class_names[index]
                predictions.append(class_name)
            else:
                predictions.append("Unknown")

        return jsonify({
            "status": "uploaded",
            "predictions": predictions
        })

@app.route('/final_result')
def final_result():
    global uploaded_files
    if not uploaded_files:
        return render_template('result.html', predictions=[])

    predictions = []
    for file_path in uploaded_files:
        data = preprocess_image(file_path)
        prediction = model.predict(data)

        # Get the predicted class index and name
        index = np.argmax(prediction)
        if 0 <= index < len(class_names):
            class_name = class_names[index]
            predictions.append(class_name)
        else:
            predictions.append("Unknown")

    uploaded_files = []  # Clear uploaded files for next submission
    return render_template('result.html', predictions=predictions)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
