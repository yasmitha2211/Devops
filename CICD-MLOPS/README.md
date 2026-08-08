# Lab-05: MLOps CI/CD Pipeline using GitHub Actions and Hugging Face

## Project Overview

This project demonstrates an end-to-end MLOps pipeline for a machine learning model using GitHub Actions and Hugging Face Hub.

The pipeline automatically prepares the data, trains the model, evaluates its performance, and deploys the trained model to Hugging Face if it satisfies the required quality threshold.

---

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- PyYAML
- Pytest
- Git & GitHub
- GitHub Actions
- Hugging Face Hub

---

## Project Structure

```text
CICD-MLOPS/
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml
│
├── data/
│   └── HousingData.csv
│
├── src/
│   ├── prepare.py
│   ├── train.py
│   ├── evaluate.py
│   └── register.py
│
├── tests/
│   └── test_pipeline.py
│
├── models/
├── params.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Workflow

```
Push to GitHub
        │
        ▼
Install Dependencies
        │
        ▼
Run Unit Tests
        │
        ▼
Prepare Dataset
        │
        ▼
Train Model
        │
        ▼
Evaluate Model
        │
        ▼
Quality Gate Check
        │
        ▼
Deploy Model to Hugging Face
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
cd CICD-MLOPS
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## Run the Project

Prepare the dataset

```bash
python src/prepare.py
```

Train the model

```bash
python src/train.py
```

Evaluate the model

```bash
python src/evaluate.py
```

Run unit tests

```bash
pytest
```

---

## GitHub Actions Pipeline

The workflow automatically performs the following steps whenever code is pushed to the `main` branch:

- Install project dependencies
- Execute unit tests
- Prepare the dataset
- Train the machine learning model
- Evaluate model performance
- Verify the quality gate
- Deploy the model to Hugging Face Hub

---

## Quality Gate

The minimum acceptable model performance is defined in `params.yaml`.

If the model score is below the threshold, deployment is automatically stopped.

---

Updated README