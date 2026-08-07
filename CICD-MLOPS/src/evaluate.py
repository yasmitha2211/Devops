import json
import joblib
import yaml
import pandas as pd
from sklearn.metrics import r2_score

# Load parameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Load test data
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# Load trained model
model = joblib.load("models/model.pkl")

# Predict
y_pred = model.predict(X_test)

# Calculate R² Score
r2 = r2_score(y_test, y_pred)

print(f"R² Score: {r2:.4f}")

# Save metrics
metrics = {
    "r2_score": float(r2)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

# Quality Gate
if r2 < params["evaluate"]["min_r2_score"]:
    raise Exception(
        f"Model failed quality gate. R² Score {r2:.4f} is below the required threshold."
    )

print("Model passed the quality gate!")