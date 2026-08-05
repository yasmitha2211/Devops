"""
Stage 4: Model Building

Trains RandomForest Regression model.

Output:
    model.pkl
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor


def train_model():

    # Load train data
    X_train = pd.read_csv("data/features/X_train.csv")
    y_train = pd.read_csv("data/features/y_train.csv")

    # Convert y dataframe to series
    y_train = y_train.values.ravel()

    # Create regression model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    print("Model training completed")

    return model


def save_model(model):

    joblib.dump(model, "model.pkl")

    print("Saved model -> model.pkl")


def main():

    model = train_model()

    save_model(model)


if __name__ == "__main__":
    main()