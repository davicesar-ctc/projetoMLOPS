import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Literal
from mlflow.tracking import MlflowClient

from src.preprocessor import DataPreprocessor

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.10"))
REGISTRY_NAME = "fraud_detection_champion"
ALIAS = "champion"

# Tradução PT → EN para os valores que o modelo conhece
_TIPO_MAP     = {"Pagamento": "Payment", "Transferência": "Transfer", "Saque": "Withdrawal"}
_MEIO_MAP     = {"Cartão": "Card", "Banco Online": "NetBanking", "PIX": "UPI", "Carteira Digital": "Wallet"}
_DEVICE_MAP   = {"Android": "Android", "iOS": "iOS", "Navegador": "Web"}
_LOCAL_MAP    = {
    "São Paulo":      "Bangalore",
    "Rio de Janeiro": "Chennai",
    "Brasília":       "Delhi",
    "Belo Horizonte": "Hyderabad",
    "Salvador":       "Mumbai",
}

_state: dict = {}
_dp = DataPreprocessor(processed_path="/tmp/api_prep")


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print(f"Conectando ao MLflow em {MLFLOW_TRACKING_URI} ...")
    _state["model"] = mlflow.sklearn.load_model(f"models:/{REGISTRY_NAME}@{ALIAS}")

    version_info = client.get_model_version_by_alias(REGISTRY_NAME, ALIAS)
    run_id = version_info.run_id
    local_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/preprocessor/preprocessor.joblib")
    _state["preprocessor"] = joblib.load(local_path)
    _state["run_id"] = run_id

    print(f"Modelo '{REGISTRY_NAME}@{ALIAS}' carregado (run_id={run_id})")
    yield
    _state.clear()


app = FastAPI(title="API de Detecção de Fraudes em Pagamentos", version="1.0.0", lifespan=lifespan)


class TransacaoInput(BaseModel):
    valor_transacao: float = Field(
        ..., description="Valor da transação em reais", example=4500.00)
    tipo_transacao: Literal["Pagamento", "Transferência", "Saque"] = Field(
        ..., description="Tipo da transação realizada", example="Pagamento")
    meio_pagamento: Literal["Cartão", "Banco Online", "PIX", "Carteira Digital"] = Field(
        ..., description="Meio de pagamento utilizado", example="Cartão")
    tipo_dispositivo: Literal["Android", "iOS", "Navegador"] = Field(
        ..., description="Dispositivo usado para realizar a transação", example="Android")
    localizacao: Literal["São Paulo", "Rio de Janeiro", "Brasília", "Belo Horizonte", "Salvador"] = Field(
        ..., description="Cidade onde a transação foi realizada", example="São Paulo")
    idade_conta_dias: int = Field(
        ..., ge=10, le=1999, description="Há quantos dias a conta do usuário existe", example=15)
    hora_transacao: int = Field(
        ..., ge=0, le=23, description="Hora do dia em que ocorreu a transação (0 = meia-noite, 23 = 23h)", example=3)
    tentativas_falhas: int = Field(
        ..., ge=0, le=4, description="Quantas tentativas de pagamento falharam antes desta", example=3)
    valor_medio_transacoes: float = Field(
        ..., description="Valor médio das transações anteriores do usuário", example=150.00)
    transacao_internacional: Literal[0, 1] = Field(
        ..., description="A transação cruzou fronteiras? 0 = Não  |  1 = Sim", example=1)
    risco_ip: float = Field(
        ..., ge=0.0, le=1.0, description="Score de risco do endereço IP (0.0 = seguro → 1.0 = muito suspeito)", example=0.92)
    tentativas_login_24h: int = Field(
        ..., ge=1, le=9, description="Número de tentativas de login nas últimas 24 horas", example=7)

    model_config = {
        "json_schema_extra": {
            "example": {
                "valor_transacao": 4500.00,
                "tipo_transacao": "Pagamento",
                "meio_pagamento": "Cartão",
                "tipo_dispositivo": "Android",
                "localizacao": "São Paulo",
                "idade_conta_dias": 15,
                "hora_transacao": 3,
                "tentativas_falhas": 3,
                "valor_medio_transacoes": 150.00,
                "transacao_internacional": 1,
                "risco_ip": 0.92,
                "tentativas_login_24h": 7,
            }
        }
    }


@app.get("/health")
def health():
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    return {
        "status": "ok",
        "modelo": f"{REGISTRY_NAME}@{ALIAS}",
        "run_id": _state.get("run_id"),
        "threshold_fraude": FRAUD_THRESHOLD,
    }


@app.post("/predict")
def predict(transacao: TransacaoInput):
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    # Converte campos PT → EN antes de passar ao preprocessor
    data = {
        "transaction_amount":       transacao.valor_transacao,
        "transaction_type":         _TIPO_MAP[transacao.tipo_transacao],
        "payment_mode":             _MEIO_MAP[transacao.meio_pagamento],
        "device_type":              _DEVICE_MAP[transacao.tipo_dispositivo],
        "device_location":          _LOCAL_MAP[transacao.localizacao],
        "account_age_days":         transacao.idade_conta_dias,
        "transaction_hour":         transacao.hora_transacao,
        "previous_failed_attempts": transacao.tentativas_falhas,
        "avg_transaction_amount":   transacao.valor_medio_transacoes,
        "is_international":         transacao.transacao_internacional,
        "ip_risk_score":            transacao.risco_ip,
        "login_attempts_last_24h":  transacao.tentativas_login_24h,
    }

    df = pd.DataFrame([data])
    df_clean = _dp.clean_data(df)
    df_feat  = _dp.create_features(df_clean)

    X    = _state["preprocessor"].transform(df_feat)
    prob = float(_state["model"].predict_proba(X)[:, 1][0])

    return {
        "fraude_detectada": prob >= FRAUD_THRESHOLD,
        "probabilidade":    round(prob, 4),
        "threshold":        FRAUD_THRESHOLD,
    }
