import os

def test_processed_data_exists():
    assert os.path.exists("data/processed/X_train.csv")
    assert os.path.exists("data/processed/X_test.csv")
    assert os.path.exists("data/processed/y_train.csv")
    assert os.path.exists("data/processed/y_test.csv")

def test_model_exists():
    assert os.path.exists("models/model.pkl")

def test_metrics_exists():
    assert os.path.exists("metrics.json")