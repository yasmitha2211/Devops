"""
Stage 1: Data Ingestion

Loads the Housing Price dataset and saves it as a raw CSV.

Output:
    data/raw/data.csv
"""

import os
import pandas as pd


def load_data() -> pd.DataFrame:
    """Load the Housing dataset into a DataFrame."""
    df = pd.read_csv(r"C:\Users\Admin\Downloads\HousePrice_DVC\HousingData.csv")
    return df


def save_raw_data(df: pd.DataFrame, out_dir: str = "data/raw") -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "data.csv")
    df.to_csv(out_path, index=False)
    print(f"[data_ingestion] Saved raw data -> {out_path} (shape={df.shape})")


def main():
    df = load_data()
    save_raw_data(df)


if __name__ == "__main__":
    main()