# 🍷 Wine Quality Prediction MLOps Project

## 📌 Project Overview

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for predicting wine quality using machine learning models. The project demonstrates dataset versioning with DVC, experiment tracking with MLflow, model deployment using FastAPI, and containerization using Docker.

---

## 🎯 Objectives

- Predict wine quality using machine learning.
- Compare multiple classification models.
- Track experiments using MLflow.
- Version datasets using DVC.
- Deploy the trained model with FastAPI.
- Containerize the application using Docker.
- Automate the workflow using GitHub Actions.

---

## 📂 Project Structure

```
Wine_Quality_MLOps/
│
├── .dvc/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── winequality-red.csv
├── mlruns/
├── models/
│   └── best_model.pkl
├── src/
│   ├── app.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── tests/
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── mlflow.db
└── README.md
```

---

## 🛠 Technologies Used

- Python 3.14
- Scikit-learn
- Pandas
- NumPy
- MLflow
- DVC
- FastAPI
- Uvicorn
- Docker
- Git
- GitHub Actions

---

## 📊 Dataset

Dataset: **Wine Quality (Red Wine)**

Target Variable:

- quality

Input Features:

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

---

## 🤖 Machine Learning Models

Three classification models were trained and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

### Best Model

Random Forest Classifier

Accuracy:

**65.94%**

---

## 🔬 Experiment Tracking

MLflow is used to:

- Track experiments
- Log model parameters
- Log evaluation metrics
- Save trained models
- Register the best model

Launch MLflow:

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

## 📦 Dataset Versioning

Initialize DVC

```bash
dvc init
```

Add Dataset

```bash
dvc add data/winequality-red.csv
```

Check Status

```bash
dvc status
```

---

## 🚀 Model Training

Run

```bash
python src/train.py
```

The script:

- Loads dataset
- Splits train/test data
- Trains three models
- Compares accuracy
- Logs experiments using MLflow
- Saves the best model

---

## 🌐 FastAPI Deployment

Run

```bash
uvicorn src.app:app --reload
```

API

```
http://127.0.0.1:8000
```

Swagger Documentation

```
http://127.0.0.1:8000/docs
```

Example Prediction

```json
{
  "predicted_quality": 4
}
```

---

## 🐳 Docker

Build Docker Image

```bash
docker build -t wine-quality-api .
```

Run Container

```bash
docker run -p 8000:8000 wine-quality-api
```

---

## ⚙ GitHub Actions

CI pipeline automatically:

- Checkout repository
- Install dependencies
- Build Docker image

Workflow File

```
.github/workflows/ci.yml
```

---

## 📈 Results

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 57.50% |
| Decision Tree | 56.25% |
| Random Forest | **65.94%** |

---

## 📷 Screenshots

Include screenshots of:

- DVC initialization
- MLflow experiments
- Registered model
- FastAPI Swagger UI
- Prediction output
- Docker build
- GitHub Actions workflow

---

## ▶️ How to Run

Clone repository

```bash
git clone <repository-url>
```

Move into project

```bash
cd Wine_Quality_MLOps
```

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python src/train.py
```

Start API

```bash
uvicorn src.app:app --reload
```

---

## 👩‍💻 Author

**Yasmithasri B**

B.Tech Information Technology

SSN College of Engineering

---

## 📄 License

This project is developed for academic and educational purposes.