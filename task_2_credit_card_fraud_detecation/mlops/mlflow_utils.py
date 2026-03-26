import mlflow
import mlflow.catboost
from task_2_credit_card_fraud_detecation.utils.utility import MODEL_NAME, MLFLOW_TRACKING_URI


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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




def register_candidate_model(candidate_run_id):
    
    model_uri = f"runs:/{candidate_run_id}/model"
    result = mlflow.register_model( model_uri=model_uri, name=MODEL_NAME)

    return { "model_name": MODEL_NAME,"version": result.version,"run_id": candidate_run_id,}