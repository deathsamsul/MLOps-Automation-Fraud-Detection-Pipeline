import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from task_2_credit_card_fraud_detecation.features.data_processing import load_and_preprocess_data
from task_2_credit_card_fraud_detecation.utils.utility import MODEL_NAME, TEST_DATA_PATH





def evaluate_model_version(version: str = "Production"):

    client = mlflow.MlflowClient()
    if version == "Production":

        # Get latest production model version
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            raise ValueError("No production model found")
        
        model_version = versions[0].version
        model_uri = f"models:/{MODEL_NAME}/{model_version}"

    else:
        model_uri = f"models:/{MODEL_NAME}/{version}"

    model = mlflow.catboost.load_model(model_uri)


    # Load holdout test set (or separate evaluation data)
    x_test, y_test = load_and_preprocess_data(TEST_DATA_PATH)  

    preds = model.predict(x_test)
    pred_proba = model.predict_proba(x_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, pred_proba),
        "f1": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds)
    }

    print(f"Evaluation for model {model_uri}:")
    
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics