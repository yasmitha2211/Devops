"""
train.py - Classification Project (Breast Cancer)
====================================================
What this file does, step by step:
 1. Load the built-in sklearn breast cancer dataset (30 numeric features)
 2. Split into train/test sets
 3. Build a Pipeline: StandardScaler -> LogisticRegression
    (Pipeline = preprocessing + model bundled together, so we only need to
    save ONE file. No separate scaler.pkl needed.)
 4. Train it
 5. Evaluate it (accuracy, precision, recall, f1)
 6. Save the trained pipeline to model.pkl using joblib

Why StandardScaler + LogisticRegression?
 - Logistic Regression IS sensitive to feature scale (unlike Random Forest),
   so we scale features first.
 - Wrapping both in a single sklearn Pipeline means predict() and
   predict_proba() automatically apply the same scaling used in training -
   no risk of forgetting to scale in the Flask app.
"""

import json
import joblib
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RANDOM_STATE = 42
TEST_SIZE = 0.2

# -----------------------------------------------------------------------
# STEP 1: Load dataset
# -----------------------------------------------------------------------
print("Step 1: Loading breast cancer dataset...")
data = load_breast_cancer()

X = data.data                     # shape (569, 30)
y = data.target                   # 0 = malignant, 1 = benign
FEATURE_NAMES = list(data.feature_names)
CLASS_NAMES = list(data.target_names)   # ['malignant', 'benign']

print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"  Classes: {CLASS_NAMES}")

# -----------------------------------------------------------------------
# STEP 2: Train/test split
# -----------------------------------------------------------------------
print("Step 2: Train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train shape: {X_train.shape}  Test shape: {X_test.shape}")

# -----------------------------------------------------------------------
# STEP 3: Build pipeline (scaler + model)
# -----------------------------------------------------------------------
print("Step 3: Building pipeline (StandardScaler + LogisticRegression)...")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
])

# -----------------------------------------------------------------------
# STEP 4: Train
# -----------------------------------------------------------------------
print("Step 4: Training...")
pipeline.fit(X_train, y_train)

# -----------------------------------------------------------------------
# STEP 5: Evaluate
# -----------------------------------------------------------------------
print("Step 5: Evaluating...")
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")

# -----------------------------------------------------------------------
# STEP 6: Save the pipeline (scaler + model together)
# -----------------------------------------------------------------------
print("Step 6: Saving pipeline to model.pkl...")
joblib.dump(pipeline, "model.pkl")
print("  Saved model.pkl")

# Save class names too, so app.py knows how to map 0/1 -> labels
with open("class_names.json", "w") as f:
    json.dump(CLASS_NAMES, f)
print("  Saved class_names.json")

metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("  Saved metrics.json")

print("\nDone. Training complete.")
