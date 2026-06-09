import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import mlflow
from scipy.sparse import load_npz
from src.model_trainer import ModelTrainer
from src.config import PROCESSED_DATA_PATH, MODELS_PATH, MODEL_NAME, METRICS_PATH

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(MODEL_NAME)

    print("A iniciar stage: TRAIN")
    
    # Carregar matrizes do disco
    print(f"A carregar artefatos de: {PROCESSED_DATA_PATH}")
    X_train = load_npz(PROCESSED_DATA_PATH / "X_train.npz")
    X_test = load_npz(PROCESSED_DATA_PATH / "X_test.npz")
    y_train = np.load(PROCESSED_DATA_PATH / "y_train.npy")
    y_test = np.load(PROCESSED_DATA_PATH / "y_test.npy")

    # Treinar modelos
    trainer = ModelTrainer(MODELS_PATH)
    results = trainer.train_models(
        X_train, y_train, X_test, y_test,
        params=params,
        experiment_name=MODEL_NAME,
        preprocessor_path=PROCESSED_DATA_PATH / "preprocessor.joblib",
    )

    # Guardar métricas em metrics/scores.json (rastreado pelo DVC)
    METRICS_PATH.mkdir(exist_ok=True)
    scores = {
        name: {k: round(v, 4) for k, v in r["metrics"].items()}
        for name, r in results.items()
    }
    
    with open(METRICS_PATH / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)
        
    print(f"Métricas guardadas com sucesso em {METRICS_PATH / 'scores.json'}")

if __name__ == "__main__":
    main()