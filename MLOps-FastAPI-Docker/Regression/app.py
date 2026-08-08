"""
app.py - Flask API for the Regression project (Boston Housing)
================================================================
What this file does:
 1. Loads model.pkl ONCE when the app starts (not on every request - that
    would be slow).
 2. Loads schema.json to know the expected feature order and count.
 3. Exposes 3 endpoints:
      GET  /        -> simple "is it running" message
      GET  /health  -> health check (used by Docker/monitoring)
      POST /predict -> takes JSON {"features": [...]} and returns a prediction
"""

import json
import joblib
import numpy as np
from flask import Flask, request, jsonify

# -----------------------------------------------------------------------
# Load model + schema ONCE at startup (not inside the /predict function)
# -----------------------------------------------------------------------
model = joblib.load("model.pkl")

with open("schema.json") as f:
    SCHEMA = json.load(f)

FEATURE_ORDER = SCHEMA["feature_order"]
NUM_FEATURES = len(FEATURE_ORDER)

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

    # --- Basic input validation ---
    if data is None or "features" not in data:
        return jsonify({
            "error": "Request body must be JSON with a 'features' key.",
            "expected_format": {"features": FEATURE_ORDER}
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
    prediction = model.predict(features_array)[0]

    return jsonify({"prediction": round(float(prediction), 2)})


if __name__ == "__main__":
    # 0.0.0.0 so the server is reachable from outside the Docker container
    app.run(host="0.0.0.0", port=8000)
