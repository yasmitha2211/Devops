# Lab-04 — ML Model Deployment (Regression + Classification)

Two complete ML projects, each trained, saved with `joblib`, and served
through a Flask API in a Docker container.

```
Lab-04/
├── Regression/        Boston Housing price prediction (RandomForestRegressor)
├── Classification/    Breast cancer diagnosis (StandardScaler + LogisticRegression)
└── README.md          This file
```

Each subfolder is self-contained: its own `train.py`, `app.py`,
`schema.json`, `model.pkl`, `requirements.txt`, `Dockerfile`,
`.dockerignore`, and `README.md` with full run/build/push instructions.

## Quick start (either project)

```bash
cd Regression        # or Classification
pip install -r requirements.txt
python train.py      # creates model.pkl
python app.py         # starts API on http://localhost:8000
```

## Docker (either project)

```bash
docker build -t <dockerhub_username>/<image_name>:latest .
docker run -p 8000:8000 <dockerhub_username>/<image_name>:latest
```

## Push to GitHub

```bash
cd Lab-04
git init
git add .
git commit -m "Lab-04: Regression + Classification ML projects with Flask + Docker"
git branch -M main
git remote add origin https://github.com/<your_username>/<repo_name>.git
git push -u origin main
```

See each project's own `README.md` for detailed step-by-step instructions,
API docs, and sample requests.
