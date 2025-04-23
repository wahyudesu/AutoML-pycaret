from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pycaret.clustering import load_model, predict_model

app = FastAPI(title="Wine Clustering API")

# Load model PyCaret saat startup
model = load_model("best_model_clustering")

# Skema input data
class WineFeatures(BaseModel):
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
    wine_type: str

@app.post("/predict")
def predict_cluster(features: WineFeatures):
    # Convert input ke dataframe
    input_df = pd.DataFrame([features.dict()])
    input_df['type_red'] = 1 if features.wine_type == 'red' else 0
    input_df['type_white'] = 0 if features.wine_type == 'red' else 1
    input_df = input_df.drop(columns=['wine_type'])

    # Prediksi cluster
    prediction = predict_model(model, data=input_df)
    cluster = prediction['Cluster'].iloc[0]

    return {
        "cluster": int(cluster)
    }

# Input

# {
#   "fixed_acidity": 7.0,
#   "volatile_acidity": 0.6,
#   "citric_acid": 0.2,
#   "residual_sugar": 2.0,
#   "chlorides": 0.08,
#   "free_sulfur_dioxide": 15.0,
#   "total_sulfur_dioxide": 45.0,
#   "density": 0.997,
#   "pH": 3.3,
#   "sulphates": 0.65,
#   "alcohol": 10.5,
#   "wine_type": "red"
# }
