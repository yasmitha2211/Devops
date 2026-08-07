"""
Stage 5: Model Evaluation

Evaluates the trained regression model.

Output:
    metrics.json
"""

import json
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model():

    # Load test data
    X_test = pd.read_csv("data/features/X_test.csv")
    y_test = pd.read_csv("data/features/y_test.csv")

    y_test = y_test.values.ravel()

    # Load trained model
    model = joblib.load("model.pkl")

    # Prediction
    y_pred = model.predict(X_test)

    # Regression metrics
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2_Score": r2_score(y_test, y_pred)
    }

    return metrics


def save_metrics(metrics):

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved metrics -> metrics.json")
    print(json.dumps(metrics, indent=4))


def main():

    metrics = evaluate_model()

    save_metrics(metrics)


if __name__ == "__main__":
    main()