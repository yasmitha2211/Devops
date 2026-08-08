import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# Load parameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Read dataset
dataset_path = params["data"]["dataset_path"]
df = pd.read_csv(dataset_path)

# Remove missing values
df = df.dropna()

# Features and target
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["data"]["test_size"],
    random_state=params["data"]["random_state"]
)

# Create processed data folder
os.makedirs("data/processed", exist_ok=True)

# Save processed files
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Data preparation completed successfully!")