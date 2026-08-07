# Classification Project — Breast Cancer Diagnosis

Predicts whether a tumor is **malignant** or **benign** from 30 numeric
measurements, using `sklearn.datasets.load_breast_cancer()` and a
StandardScaler + LogisticRegression pipeline served through Flask.

## Files

| File | Purpose |
|---|---|
| `train.py` | Loads dataset, trains the pipeline, saves `model.pkl` |
| `app.py` | Flask API that loads `model.pkl` and serves predictions |
| `schema.json` | Describes the 30 expected input features, in order |
| `model.pkl` | Trained pipeline (scaler + classifier), created by `train.py` |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Builds a container image for the API |
| `.dockerignore` | Files excluded from the Docker build context |

Note: this project doesn't need a separate CSV file — the dataset is
built into scikit-learn (`load_breast_cancer()`), so `train.py` downloads
nothing and works offline.

## 1. Train the model

```bash
cd Classification
pip install -r requirements.txt
python train.py
```

**Expected output:** training logs ending in `Done. Training complete.`,
with accuracy around 0.97–0.99. `model.pkl` is created.

## 2. Verify model.pkl was created

```bash
ls -la model.pkl
```

## 3. Run the API locally

```bash
python app.py
```

**Expected output:** `Running on http://0.0.0.0:8000`

## 4. Test the API

### Using curl
```bash
curl http://localhost:8000/
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'
```

Expected response:
```json
{"predicted_class":"malignant","confidence":1.0,"probabilities":{"malignant":1.0,"benign":0.0}}
```

### Using Postman
1. Method: `POST`
2. URL: `http://localhost:8000/predict`
3. Body → raw → JSON: a `features` array with 30 numbers (see `schema.json`
   for the exact order and an example).
4. Send → you get `predicted_class`, `confidence`, and `probabilities`.

## 5. Build the Docker image

```bash
docker build -t <dockerhub_username>/classification-cancer:latest .
```

## 6. Run the container

```bash
docker run -p 8000:8000 <dockerhub_username>/classification-cancer:latest
```

## 7. Test the container

Same curl commands as step 4.

## 8. Docker login / push / pull

```bash
docker login
docker push <dockerhub_username>/classification-cancer:latest

# On another machine:
docker pull <dockerhub_username>/classification-cancer:latest
docker run -p 8000:8000 <dockerhub_username>/classification-cancer:latest
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Returns `{"message":"Model API Running"}` |
| GET | `/health` | Returns `{"status":"ok"}` |
| POST | `/predict` | Takes `{"features":[...30 numbers...]}`, returns `predicted_class`, `confidence`, `probabilities` |

See `schema.json` for the full list of 30 feature names, in the exact
order the API expects them.
