"""
app.py - Flask API for the Classification project (Breast Cancer)
====================================================================
Same structure as the Regression app.py:
 1. Load model.pkl ONCE at startup (it's a full sklearn Pipeline:
    StandardScaler + LogisticRegression, so no separate scaler file needed).
 2. Load schema.json for feature order/count, and class name mapping.
 3. Expose GET /, GET /health, POST /predict.
"""

import json
import joblib
import numpy as np
from flask import Flask, request, jsonify

# -----------------------------------------------------------------------
# Load model + schema ONCE at startup
# -----------------------------------------------------------------------
model = joblib.load("model.pkl")   # this is a Pipeline (scaler + classifier)

with open("schema.json") as f:
    SCHEMA = json.load(f)

FEATURE_ORDER = SCHEMA["feature_order"]
NUM_FEATURES = len(FEATURE_ORDER)

# sklearn's load_breast_cancer encodes: 0 = malignant, 1 = benign
# model.classes_ gives us the order sklearn actually uses internally,
# so we map by that instead of hardcoding, in case it ever changes.
CLASS_NAMES = {0: "malignant", 1: "benign"}

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Model API Running"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if data is None or "features" not in data:
        return jsonify({
            "error": "Request body must be JSON with a 'features' key.",
            "expected_num_features": NUM_FEATURES
        }), 400

    features = data["features"]

    if not isinstance(features, list):
        return jsonify({"error": "'features' must be a list of numbers."}), 400

    if len(features) != NUM_FEATURES:
        return jsonify({
            "error": f"Expected {NUM_FEATURES} features, got {len(features)}.",
            "feature_order": FEATURE_ORDER
        }), 400

    try:
        features_array = np.array(features, dtype=float).reshape(1, -1)
    except (ValueError, TypeError):
        return jsonify({"error": "All features must be numeric."}), 400

    # --- Predict ---
    predicted_label = int(model.predict(features_array)[0])
    probabilities = model.predict_proba(features_array)[0]

    predicted_class = CLASS_NAMES[predicted_label]
    prob_dict = {
        CLASS_NAMES[i]: round(float(probabilities[i]), 4)
        for i in range(len(probabilities))
    }

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": round(float(max(probabilities)), 4),
        "probabilities": prob_dict
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
