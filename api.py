import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.context import PipelineContext

app = FastAPI(
    title="Motor de Inferência - Hotel Bookings", 
    description="API de Produção para prever o cancelamento de reservas hoteleiras.",
    version="1.0"
)

context = PipelineContext.from_notebook(__file__)

def load_champion_model():
    print("A carregar o modelo do MLflow...")
    mlflow_uri = str(context.get_path("mlruns.db"))
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_uri}")
    
    experiment = mlflow.get_experiment_by_name("hotel_cancellation_prediction")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'RandomForest_Baseline'",
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )
    
    if runs.empty:
        raise Exception("Modelo Baseline não encontrado no banco do MLflow.")
        
    best_run_id = runs.iloc[0].run_id
    model_uri = f"runs:/{best_run_id}/model"
    
    return mlflow.sklearn.load_model(model_uri)

modelo_producao = load_champion_model()

class BookingRequest(BaseModel):
    features: List[Dict[str, Any]]

@app.post("/predict")
def predict(request: BookingRequest):
    try:
        df_novos_hospedes = pd.DataFrame(request.features)

        predicoes = modelo_producao.predict(df_novos_hospedes)
        probabilidades = modelo_producao.predict_proba(df_novos_hospedes)[:, 1]
        
        resultados = []
        for pred, prob in zip(predicoes, probabilidades):
            resultados.append({
                "is_canceled_prediction": int(pred),
                "probability_of_cancellation": round(float(prob), 4),
                "risk_level": "Alto Risco" if prob > 0.6 else "Baixo Risco"
            })
            
        return {"status": "success", "predictions": resultados}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro no processamento: {str(e)}")