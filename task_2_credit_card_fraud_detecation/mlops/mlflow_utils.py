import mlflow
import mlflow.catboost
from task_2_credit_card_fraud_detecation.utils.utility import MODEL_NAME




#load form model registry
def load_production_model():

    client = mlflow.MlflowClient()
    try:
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not latest_versions:
            raise RuntimeError("No model in Production stage")
        model_version = latest_versions[0].version
        model_uri = f"models:/{MODEL_NAME}/{model_version}"  # model registry URI ,models:/fraud_detection_model/3
        model = mlflow.catboost.load_model(model_uri)    # catboost model loader helper functon
        return model
    
    except Exception as e:
        print(f"Error loading production model: {e}")
        raise


def log_model_to_registry(model, run_id, metrics, model_name=MODEL_NAME):
    with mlflow.start_run(run_id=run_id):
        mlflow.catboost.log_model(model, "model", registered_model_name=model_name)
        mlflow.log_metrics(metrics)