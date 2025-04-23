from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.classification import load_model, predict_model
import pandas as pd

# Inisialisasi FastAPI
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API untuk memprediksi kemungkinan kanker payudara berulang menggunakan model PyCaret",
    version="1.0.0"
)

# Load model PyCaret
model = load_model("best_model_classification")

# Skema input data dari client
class CancerInput(BaseModel):
    age: str
    menopause: str
    tumor_size: str
    inv_nodes: str
    node_caps: str
    deg_malig: int
    breast: str
    breast_quad: str
    irradiat: str

# Endpoint prediksi
@app.post("/predict")
def predict_cancer(data: CancerInput):
    input_df = pd.DataFrame([data.dict()])

    try:
        result = predict_model(model, data=input_df)
        pred = result.loc[0, 'prediction_label']
        prob = result.loc[0, 'prediction_score']

        if pred == 'no-recurrence-events':
            status = "🟢 Tidak ada kanker berulang"
        else:
            status = "🔴 Terindikasi kanker berulang"

        return {
            "prediction": pred,
            "probability": round(prob, 4),
            "status": status
        }
    except Exception as e:
        return {"error": str(e)}

# Input
#
# {
#   "age": "50-59",
#   "menopause": "ge40",
#   "tumor_size": "15-19",
#   "inv_nodes": "0-2",
#   "node_caps": "no",
#   "deg_malig": 2,
#   "breast": "left",
#   "breast_quad": "left_up",
#   "irradiat": "yes"
# }
