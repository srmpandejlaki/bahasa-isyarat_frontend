from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import json
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Load model dan class mapping
model = load_model('model/training_results/bisindo_model.h5')
with open('model/training_results/class_indices.json') as f:
    class_map = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(BytesIO(file.read())).convert("RGB")
    img = img.resize((128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    predicted_class = class_map[str(predicted_index)]

    return jsonify({
        'prediction': predicted_class,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
