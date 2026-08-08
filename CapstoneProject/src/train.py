import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/winequality-red.csv", sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Models
# -----------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42)
}

os.makedirs("models", exist_ok=True)

best_accuracy = 0
best_model = None
best_name = ""

mlflow.set_experiment("Wine_Quality_Classification")

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model"
        )

        print(f"{name}: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline
            best_name = name

joblib.dump(best_model, "models/best_model.pkl")

print("\nBest Model:", best_name)
print("Accuracy:", round(best_accuracy,4))