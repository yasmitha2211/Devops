"""
train.py - Regression Project (Boston Housing)
================================================
What this file does, step by step:
 1. Load HousingData.csv
 2. Clean the data (drop duplicates, fill missing values with median)
 3. Split into features (X) and target (y)
 4. Split into train/test sets
 5. Train a RandomForestRegressor
 6. Evaluate it (MAE, MSE, RMSE, R2)
 7. Save the trained model to model.pkl using joblib

Why RandomForestRegressor?
 - It does NOT need feature scaling (unlike Linear Regression), so we don't
   need to save/load a separate scaler.pkl file. This keeps the Flask API
   simple: load one file, predict.
 - It generally performs well on tabular data like this without much tuning.
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
TEST_SIZE = 0.2

# -----------------------------------------------------------------------
# STEP 1: Load the dataset
# -----------------------------------------------------------------------
print("Step 1: Loading dataset...")
raw_df = pd.read_csv("HousingData.csv")
print(f"  Raw shape: {raw_df.shape}")

# -----------------------------------------------------------------------
# STEP 2: Clean the data
# -----------------------------------------------------------------------
print("Step 2: Cleaning data...")

# Standardize column names (strip spaces, no accidental whitespace issues)
processed_df = raw_df.copy()
processed_df.columns = [c.strip().replace(" ", "_") for c in processed_df.columns]

# Remove exact duplicate rows
before = processed_df.shape[0]
processed_df = processed_df.drop_duplicates()
print(f"  Dropped {before - processed_df.shape[0]} duplicate rows")

# HousingData.csv has some missing values (NA). Fill them with the column
# median. Median is used instead of mean because it is less sensitive to
# outliers (e.g. CRIM has extreme high values in a few rows).
missing_before = processed_df.isnull().sum().sum()
print(f"  Missing values found: {missing_before}")
processed_df = processed_df.fillna(processed_df.median(numeric_only=True))

# -----------------------------------------------------------------------
# STEP 3: Split into features (X) and target (y)
# -----------------------------------------------------------------------
print("Step 3: Splitting features/target...")

TARGET_COLUMN = "MEDV"
FEATURE_COLUMNS = [c for c in processed_df.columns if c != TARGET_COLUMN]

X = processed_df[FEATURE_COLUMNS]
y = processed_df[TARGET_COLUMN]

print(f"  Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

# -----------------------------------------------------------------------
# STEP 4: Train/test split
# -----------------------------------------------------------------------
print("Step 4: Train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"  Train shape: {X_train.shape}  Test shape: {X_test.shape}")

# -----------------------------------------------------------------------
# STEP 5: Train the model
# -----------------------------------------------------------------------
print("Step 5: Training RandomForestRegressor...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)

# -----------------------------------------------------------------------
# STEP 6: Evaluate the model
# -----------------------------------------------------------------------
print("Step 6: Evaluating model...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"  MAE  : {mae:.4f}")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R2   : {r2:.4f}")

# -----------------------------------------------------------------------
# STEP 7: Save the model
# -----------------------------------------------------------------------
print("Step 7: Saving model to model.pkl...")
joblib.dump(model, "model.pkl")
print("  Saved model.pkl")

# Also save metrics to a small json file (useful later for MLflow / reports)
metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("  Saved metrics.json")

print("\nDone. Training complete.")
