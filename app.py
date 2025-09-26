from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

# --------- Definir FastAPI ---------
app = FastAPI()

# --------- Cargar modelo ----------
class MiModelo(nn.Module):
    def __init__(self):
        super(MiModelo, self).__init__()
        # ⚠️ Debes replicar la arquitectura que usaste al entrenar
        self.fc = nn.Linear(10, 2)  # <-- Ejemplo (cámbialo por tu modelo real)

    def forward(self, x):
        return self.fc(x)

# Cargar pesos
modelo = MiModelo()
# modelo.load_state_dict(torch.load("mi_modelo.pth", map_location=torch.device("cpu")))
# modelo.eval()

# --------- Esquema de entrada ---------
class InputData(BaseModel):
    features: list[float]  # Lista de floats como input del modelo

# --------- Endpoint opcional /health ---------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# --------- Endpoint /predict ---------
@app.post("/predict")
def predict(data: InputData):
    # Convertir a tensor
    x = torch.tensor(data.features, dtype=torch.float32)
    # Si el input es vector, añadir batch dimension
    if x.ndim == 1:
        x = x.unsqueeze(0)

    with torch.no_grad():
        output = modelo(x)
        pred = torch.argmax(output, dim=1).item()

    return {"prediction": pred}
