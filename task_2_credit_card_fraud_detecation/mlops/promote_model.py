import mlflow
from task_2_credit_card_fraud_detecation.utils.utility import MODEL_NAME




def promote_to_production(version: str, stage: str = "Production"):

    client = mlflow.MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage=stage
    )
    print(f"Model {MODEL_NAME} version {version} moved to {stage}.")



def archive_current_production():
    
    client = mlflow.MlflowClient()
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    for v in prod_versions:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=v.version,
            stage="Archived"
        )
        print(f"Archived version {v.version}")