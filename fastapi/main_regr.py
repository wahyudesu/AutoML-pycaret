from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.regression import load_model, predict_model
import pandas as pd

app = FastAPI(
    title="Prediksi Harga Rumah",
    description="API untuk memprediksi harga rumah menggunakan model PyCaret",
    version="1.0.0"
)

# Load model dari file .pkl
model = load_model("best_model_regression")

# Skema input data
class HouseFeatures(BaseModel):
    OverallQual: int
    GrLivArea: float
    GarageCars: int
    TotalBsmtSF: float

# Endpoint prediksi
@app.post("/predict")
def predict_price(features: HouseFeatures):
    input_df = pd.DataFrame([features.dict()])
    prediction = predict_model(model, data=input_df)
    hasil = prediction["Label"].iloc[0]
    return {"predicted_price": round(hasil, 2)}

# Input

# {
#   "OverallQual": 7,
#   "GrLivArea": 2000,
#   "GarageCars": 2,
#   "TotalBsmtSF": 1000
# }
