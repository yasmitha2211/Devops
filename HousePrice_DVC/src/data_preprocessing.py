"""
Stage 2: Data Preprocessing

Cleans the raw housing dataset:
- Removes duplicates
- Handles missing values
- Saves processed dataset

Output:
    data/processed/data.csv
"""

import os
import pandas as pd


def preprocess_data(input_path="data/raw/data.csv"):

    df = pd.read_csv(input_path)

    # Clean column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Remove duplicate rows
    before = df.shape[0]
    df = df.drop_duplicates()

    print(f"Dropped {before - df.shape[0]} duplicate rows")

    # Handle missing values
    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing}")

    if missing > 0:
        df = df.fillna(df.median(numeric_only=True))

    return df


def save_processed_data(df, out_dir="data/processed"):

    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "data.csv")

    df.to_csv(out_path, index=False)

    print(f"[data_preprocessing] Saved processed data -> {out_path} (shape={df.shape})")


def main():

    df = preprocess_data()

    save_processed_data(df)


if __name__ == "__main__":
    main()