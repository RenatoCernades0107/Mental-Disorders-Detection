from fastapi import FastAPI
from pydantic import BaseModel, constr
from typing import Literal
import pandas as pd
from sklearn import svm
import uvicorn

import torch
import torch.nn as nn
import joblib
from dataset import MDDataset
# --------- Definir FastAPI ---------
app = FastAPI()

# --------- Cargar dataset y modelo ---------
file_path = './dataset/mental_disorders_dataset.csv'  # Path to your CSV file
target_column = 'Expert Diagnose'  # Name of the target column in your CSV
dataset = MDDataset(file_path, target_column, normalize=None, pca_components=3)

modelo = svm.SVC(kernel='linear', C=1.0, random_state=42)
# Cargar pesos
try:
    modelo = joblib.load('./models/svm_model.pkl')
    print("Modelo cargado exitosamente.")
except FileNotFoundError:
    print("No se encontró el modelo.")

# --------- Esquema de entrada ---------

class InputData(BaseModel):
    # Solo se permiten estos valores
    sadness: Literal["Sometimes", "Usually", "Most-Often", "Seldom"]
    euphoric: Literal["Sometimes", "Usually", "Most-Often", "Seldom"]
    exhausted: Literal["Sometimes", "Usually", "Most-Often", "Seldom"]
    sleep_disorder: Literal["Sometimes", "Usually", "Most-Often", "Seldom"]

    # Validación Yes/No
    mood_swing: Literal["Yes", "No"]
    suicidal_thoughts: Literal["Yes", "No"]
    anorexia: Literal["Yes", "No"]
    authority_respect: Literal["Yes", "No"]
    try_explanation: Literal["Yes", "No"]

    aggressive_response: Literal["Yes", "No"]
    ignore_move_on: Literal["Yes", "No"]
    nervous_breakdown: Literal["Yes", "No"]
    admit_mistakes: Literal["Yes", "No"]
    overthinking: Literal["Yes", "No"]

    # Solo valores tipo "x From y"
    sexual_activity: str
    concentration: str
    optimism: str


# --------- Endpoint opcional /health ---------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# --------- Endpoint /predict ---------
@app.post("/predict")
def predict(data: InputData):
    # Convertir a tensor
    x = dataset.process_point(data)

    with torch.no_grad():
        output = modelo.predict(x.numpy().reshape(1, -1))
        # to tensor
        output = torch.tensor(output)
        pred = torch.argmax(output).item()

    pred_label = dataset.get_labels()[pred]
    return {"prediction": pred_label}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
