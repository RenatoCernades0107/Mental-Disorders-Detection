from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# --------- Definir FastAPI ---------
app = FastAPI()

# --------- Cargar modelo ----------
# Asegúrate que exista el archivo ./models/svm_model.pkl
model = joblib.load("./models/svm_model.pkl")

# --------- Esquema de entrada ---------
class InputData(BaseModel):
    features: list[float]  # Lista de características numéricas

# --------- Endpoint opcional /health ---------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# --------- Endpoint /predict ---------
@app.post("/predict")
def predict(data: InputData):
    # Convertir a numpy array (1 muestra = 1 fila)
    x = np.array(data.features).reshape(1, -1)
    
    # Predicción
    pred = model.predict(x)[0]  # solo un valor
    return {"prediction": int(pred)}
