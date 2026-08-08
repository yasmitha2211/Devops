import os
import joblib
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load parameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Load training data
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

# Create model
model = RandomForestRegressor(
    n_estimators=params["model"]["n_estimators"],
    random_state=params["model"]["random_state"]
)

# Train model
model.fit(X_train, y_train)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/model.pkl")

print("Model trained successfully!")