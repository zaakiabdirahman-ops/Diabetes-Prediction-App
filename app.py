import numpy as np
import pickle
import os
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, template_folder='Template', static_folder='static')

# Load model once on startup
MODEL_PATH = 'diabetes_model.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place your diabetes_model.pkl in the project root.")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Helper: safe float conversion (default 0.0 if not parseable)
def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

@app.route('/')
def home():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Accept either JSON (from fetch) or form-data (fallback)
        data = request.get_json(silent=True)
        if data is None:
            data = request.form.to_dict()

        # Read expected 6 features (order must match model training)
        age = to_float(data.get('age'))
        hypertension = to_float(data.get('hypertension'))
        heart_disease = to_float(data.get('heart_disease'))
        bmi = to_float(data.get('bmi'))
        HbA1c_level = to_float(data.get('HbA1c_level'))
        blood_glucose_level = to_float(data.get('blood_glucose_level'))

        features = np.array([[age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level]])

        # Basic shape check
        if features.shape[1] != 6:
            return jsonify({'output': f'❌ Feature count mismatch: got {features.shape[1]} but model expects 6', 'prediction': None})

        # Prediction
        prediction = None
        confidence = None

        # Try to get predict_proba if available
        try:
            proba = model.predict_proba(features)
            # assume binary classification [prob_neg, prob_pos]
            # For multiclass, pick highest class probability
            if proba.shape[1] == 2:
                confidence = float(proba[0, 1])  # probability of positive (1)
            else:
                confidence = float(np.max(proba))
        except Exception:
            # model may not support predict_proba
            confidence = None

        # Get final class
        try:
            prediction = int(model.predict(features)[0])
        except Exception as e:
            return jsonify({'output': f'❌ Model prediction error: {str(e)}', 'prediction': None})

        # Prepare friendly message
        if prediction == 1:
            message = "⚠️ The model predicts the patient *may have diabetes*. Please consult a healthcare professional."
        else:
            message = "✅ The model predicts *No Diabetes*. Maintain a healthy lifestyle."

        response = {
            'output': message,
            'prediction': prediction,
            'confidence': confidence
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'output': f'❌ An unexpected error occurred: {str(e)}', 'prediction': None})

if __name__ == "__main__":
    app.run(debug=True)
