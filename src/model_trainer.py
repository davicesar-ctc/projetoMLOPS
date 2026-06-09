import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from mlflow.tracking import MlflowClient

class ModelTrainer:
    """Classe para treinar modelos e registrar no MLflow."""

    def __init__(self, models_path: Path):
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)

    def train_models(self, X_train, y_train, X_test, y_test, params: dict, experiment_name: str,
                     preprocessor_path: Path = None):
        mlflow.set_experiment(experiment_name)
        results = {}

        # Calcula o peso para o XGBoost dinamicamente
        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        scale_pos_weight = neg / pos if pos > 0 else 1

        models = {
            'xgboost': XGBClassifier(
                n_estimators=params['xgboost']['n_estimators'],
                max_depth=params['xgboost']['max_depth'],
                learning_rate=params['xgboost']['learning_rate'],
                subsample=params['xgboost']['subsample'],
                colsample_bytree=params['xgboost']['colsample_bytree'],
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="aucpr",
                random_state=params['train']['random_state'],
                n_jobs=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=params['random_forest']['n_estimators'],
                min_samples_leaf=params['random_forest']['min_samples_leaf'],
                class_weight="balanced",
                random_state=params['train']['random_state'],
                n_jobs=-1
            )
        }

        for model_name, model in models.items():
            print(f"\nTreinando {model_name}...")

            with mlflow.start_run(run_name=model_name):
                # Treina o modelo
                model.fit(X_train, y_train)

                # Predições
                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= params['train']['threshold']).astype(int)

                # Métricas
                metrics = {
                    'roc_auc': roc_auc_score(y_test, y_prob),
                    'pr_auc': average_precision_score(y_test, y_prob),
                    'accuracy': accuracy_score(y_test, y_pred)
                }

                # Logar no MLflow
                mlflow.log_param("model_type", model_name)
                if model_name == 'xgboost':
                    mlflow.log_param("scale_pos_weight", scale_pos_weight)

                mlflow.log_params(params.get(model_name, {}))
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, model_name)

                if preprocessor_path and Path(preprocessor_path).exists():
                    mlflow.log_artifact(str(preprocessor_path), "preprocessor")

                run_id = mlflow.active_run().info.run_id

                # Salvar o modelo em disco (.joblib)
                model_path = self.models_path / f"{model_name}.joblib"
                joblib.dump(model, model_path)

                results[model_name] = {
                    'metrics': metrics,
                    'path': model_path,
                    'run_id': run_id,
                }

        # Registrar o campeão no MLflow Model Registry
        best_name = max(results, key=lambda m: results[m]['metrics']['roc_auc'])
        best_run_id = results[best_name]['run_id']
        best_auc = results[best_name]['metrics']['roc_auc']

        print(f"\nRegistrando campeão: {best_name} (ROC-AUC: {best_auc:.4f})")
        model_uri = f"runs:/{best_run_id}/{best_name}"
        registered = mlflow.register_model(model_uri, "fraud_detection_champion")

        client = MlflowClient()
        client.set_registered_model_alias("fraud_detection_champion", "champion", registered.version)
        print(f"Alias 'champion' setado na versão {registered.version}")

        return results