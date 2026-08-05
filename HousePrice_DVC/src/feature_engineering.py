"""
Stage 3: Feature Engineering

- Separates features and target
- Splits train and test data
- Applies StandardScaler

Outputs:
    data/features/X_train.csv
    data/features/X_test.csv
    data/features/y_train.csv
    data/features/y_test.csv
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def feature_engineering(input_path="data/processed/data.csv"):

    df = pd.read_csv(input_path)

    # Change this if your target column name is different
    X = df.drop(columns=["MEDV"])
    y = df["MEDV"]

    # Train-test split (no stratify for regression)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to dataframe
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_features(X_train, X_test, y_train, y_test, scaler):

    os.makedirs("data/features", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X_train.to_csv("data/features/X_train.csv", index=False)
    X_test.to_csv("data/features/X_test.csv", index=False)

    y_train.to_csv("data/features/y_train.csv", index=False)
    y_test.to_csv("data/features/y_test.csv", index=False)

    joblib.dump(scaler, "models/scaler.pkl")

    print("[feature_engineering] Features saved successfully")


def main():

    X_train, X_test, y_train, y_test, scaler = feature_engineering()

    save_features(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )


if __name__ == "__main__":
    main()