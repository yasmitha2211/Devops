# Regression Project — Boston Housing Price Prediction

Predicts the median house value (`MEDV`, in $1000s) for a Boston-area tract
using 13 numeric features, via a RandomForestRegressor served through Flask.

## Files

| File | Purpose |
|---|---|
| `HousingData.csv` | Raw dataset |
| `train.py` | Cleans data, trains the model, saves `model.pkl` |
| `app.py` | Flask API that loads `model.pkl` and serves predictions |
| `schema.json` | Describes the expected input features and their order |
| `model.pkl` | Trained model (created by `train.py`) |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Builds a container image for the API |
| `.dockerignore` | Files excluded from the Docker build context |

## 1. Train the model

```bash
cd Regression
pip install -r requirements.txt
python train.py
```

**Expected output:** training logs ending in `Done. Training complete.`,
and R² around 0.85–0.90. Two new files appear: `model.pkl` and `metrics.json`.

## 2. Verify model.pkl was created

```bash
ls -la model.pkl
```

You should see the file with a non-zero size (a few hundred KB, since it's
a Random Forest with 100 trees).

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
  -d '{"features":[0.00632,18.0,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98]}'
```

Expected response:
```json
{"prediction": 28.35}
```

### Using Postman
1. Method: `POST`
2. URL: `http://localhost:8000/predict`
3. Body → raw → JSON:
```json
{"features":[0.00632,18.0,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98]}
```
4. Send → you should get a `prediction` field back.

## 5. Build the Docker image

```bash
docker build -t <dockerhub_username>/regression-boston:latest .
```

## 6. Run the container

```bash
docker run -p 8000:8000 <dockerhub_username>/regression-boston:latest
```

## 7. Test the container

Same curl commands as step 4 — the API behaves identically inside Docker.

## 8. Docker login / push / pull

```bash
docker login
docker push <dockerhub_username>/regression-boston:latest

# On another machine:
docker pull <dockerhub_username>/regression-boston:latest
docker run -p 8000:8000 <dockerhub_username>/regression-boston:latest
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Returns `{"message":"Model API Running"}` |
| GET | `/health` | Returns `{"status":"ok"}` |
| POST | `/predict` | Takes `{"features":[...13 numbers...]}`, returns `{"prediction": <float>}` |

Feature order (see `schema.json` for full descriptions):
`CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT`
