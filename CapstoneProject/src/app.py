from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Wine Quality Prediction API")

# Load trained model
model = joblib.load("models/best_model.pkl")

# Input schema
class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Wine Quality Prediction API is running!"
    }

# Prediction endpoint
@app.post("/predict")
def predict(data: WineInput):
    df = pd.DataFrame([{
        "fixed acidity": data.fixed_acidity,
        "volatile acidity": data.volatile_acidity,
        "citric acid": data.citric_acid,
        "residual sugar": data.residual_sugar,
        "chlorides": data.chlorides,
        "free sulfur dioxide": data.free_sulfur_dioxide,
        "total sulfur dioxide": data.total_sulfur_dioxide,
        "density": data.density,
        "pH": data.pH,
        "sulphates": data.sulphates,
        "alcohol": data.alcohol
    }])

    prediction = model.predict(df)

    return {
        "predicted_quality": int(prediction[0])
    }