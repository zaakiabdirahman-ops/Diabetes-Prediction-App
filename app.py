import numpy as np
import pickle
import os
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, template_folder='Template', static_folder='static')

# Load model once on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__),'diabetes_prediction_model .pkl')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Place your diabetes_model.pkl in the same folder as app.py.")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


# Helper: safe float conversion (default 0.0 if not parseable)
def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


# Generate feature analysis based on input values
def generate_feature_analysis(features_dict):
    analysis = []

    # HbA1c analysis (normal: <5.7%, prediabetes: 5.7-6.4%, diabetes: >6.4%)
    hba1c = features_dict['HbA1c_level']
    if hba1c > 6.4:
        analysis.append("Your HbA1c level indicates possible diabetes (normal range is below 5.7%).")
    elif hba1c > 5.7:
        analysis.append("Your HbA1c level is in the prediabetes range (5.7-6.4%).")
    else:
        analysis.append("Your HbA1c level is within the normal range (below 5.7%).")

    # Blood glucose analysis (normal fasting: <100 mg/dL, prediabetes: 100-125 mg/dL, diabetes: >125 mg/dL)
    glucose = features_dict['blood_glucose_level']
    if glucose > 125:
        analysis.append("Your blood glucose level suggests possible diabetes (normal fasting is below 100 mg/dL).")
    elif glucose > 100:
        analysis.append("Your blood glucose level is in the prediabetes range (100-125 mg/dL).")
    else:
        analysis.append("Your blood glucose level is within the normal range (below 100 mg/dL).")

    # BMI analysis (normal: 18.5-24.9, overweight: 25-29.9, obese: >30)
    bmi = features_dict['bmi']
    if bmi > 30:
        analysis.append("Your BMI indicates obesity, which is a significant risk factor for diabetes.")
    elif bmi > 25:
        analysis.append("Your BMI indicates overweight, which increases diabetes risk.")
    elif bmi < 18.5:
        analysis.append(
            "Your BMI indicates underweight. While not directly linked to diabetes, maintaining a healthy weight is important.")
    else:
        analysis.append("Your BMI is within the healthy range (18.5-24.9).")

    # Age analysis (risk increases with age, especially after 45)
    age = features_dict['age']
    if age > 45:
        analysis.append("Your age places you in a higher risk category for diabetes.")
    else:
        analysis.append("Your age is not a significant risk factor for diabetes.")

    # Hypertension and heart disease analysis
    if features_dict['hypertension'] == 1:
        analysis.append("Hypertension is associated with increased diabetes risk.")

    if features_dict['heart_disease'] == 1:
        analysis.append("Heart disease is correlated with higher diabetes risk.")

    return analysis


# Calculate risk score based on medical guidelines
def calculate_risk_score(features_dict):
    score = 0

    # HbA1c level is a strong indicator
    if features_dict['HbA1c_level'] > 6.4:
        score += 0.4
    elif features_dict['HbA1c_level'] > 5.7:
        score += 0.2

    # Blood glucose level
    if features_dict['blood_glucose_level'] > 125:
        score += 0.3
    elif features_dict['blood_glucose_level'] > 100:
        score += 0.15

    # BMI
    if features_dict['bmi'] > 30:
        score += 0.15
    elif features_dict['bmi'] > 25:
        score += 0.1

    # Age
    if features_dict['age'] > 45:
        score += 0.1

    # Hypertension and heart disease
    if features_dict['hypertension'] == 1:
        score += 0.05
    if features_dict['heart_disease'] == 1:
        score += 0.05

    return min(score, 1.0)


# Add CORS headers manually
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


@app.route('/')
def home():
    return render_template('test.html')


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200

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
        features_dict = {
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'bmi': bmi,
            'HbA1c_level': HbA1c_level,
            'blood_glucose_level': blood_glucose_level
        }

        # Basic shape check
        if features.shape[1] != 6:
            return jsonify({
                'error': f'Feature count mismatch: got {features.shape[1]} but model expects 6',
                'prediction': None
            })

        # Prediction
        prediction = None
        confidence = None

        # Try to get predict_proba if available
        try:
            proba = model.predict_proba(features)
            # assume binary classification [prob_neg, prob_pos]
            if proba.shape[1] == 2:
                confidence = float(proba[0, 1])  # probability of positive (1)
                risk_score = float(proba[0, 1])  # Use model's probability as risk score
            else:
                confidence = float(np.max(proba))
                risk_score = float(np.max(proba))
        except Exception:
            # model may not support predict_proba, use guideline-based calculation
            confidence = None
            risk_score = calculate_risk_score(features_dict)

        # Get final class
        try:
            prediction = int(model.predict(features)[0])
        except Exception as e:
            return jsonify({
                'error': f'Model prediction error: {str(e)}',
                'prediction': None
            })

        # Generate feature analysis
        feature_analysis = generate_feature_analysis(features_dict)

        # Prepare response matching frontend expectations
        response = {
            'prediction': prediction,
            'confidence': round(confidence * 100, 2) if confidence is not None else round(risk_score * 100, 2),
            'risk_score': risk_score,
            'feature_analysis': feature_analysis
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': f'An unexpected error occurred: {str(e)}',
            'prediction': None
        })


# Add a health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': True})


# Add a model info endpoint for the frontend
@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type': str(type(model).__name__),
        'features': ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level'],
        'supported_methods': {
            'predict': True,
            'predict_proba': hasattr(model, 'predict_proba')
        }
    })


if __name__ == "__main__":
    app.run(debug=True)