import mlflow
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from task_2_credit_card_fraud_detecation.features.data_processing import load_and_preprocess_data
from task_2_credit_card_fraud_detecation.utils.utility import MODEL_NAME, TEST_DATA_PATH




mlflow.set_tracking_uri("sqlite:///mlflow.db")

def calculate_metrics(model, x_test, y_test):
    
    preds = model.predict(x_test)
    pred_proba = model.predict_proba(x_test)[:, 1]

    metrics = { "roc_auc": roc_auc_score(y_test, pred_proba),"f1": f1_score(y_test, preds),
               "precision": precision_score(y_test, preds),"recall": recall_score(y_test, preds), }
    
    return metrics


def evaluate_model_version(version: str = "Production"):

    client = mlflow.MlflowClient()
    if version == "Production":
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            raise ValueError("No production model found")

        model_version = versions[0].version
        model_uri = f"models:/{MODEL_NAME}/{model_version}"
    else:
        model_uri = f"models:/{MODEL_NAME}/{version}"

    model = mlflow.catboost.load_model(model_uri)
    x_test, y_test = load_and_preprocess_data(TEST_DATA_PATH)
    metrics = calculate_metrics(model, x_test, y_test)

    print(f"Evaluation for model {model_uri}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics


def evaluate_candidate_model(candidate_run_id: str, min_roc_auc: float = 0.80, max_recall_drop: float = 0.02):
    
    client = mlflow.MlflowClient()    
    x_test, y_test = load_and_preprocess_data(TEST_DATA_PATH)

    candidate_model_uri = f"runs:/{candidate_run_id}/model"
    candidate_model = mlflow.catboost.load_model(candidate_model_uri)
    candidate_metrics = calculate_metrics(candidate_model, x_test, y_test)

    production_metrics = { "roc_auc": 0.0,"f1": 0.0,"precision": 0.0,"recall": 0.0,}

    try:
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if prod_versions:
            prod_version = prod_versions[0].version
            prod_model_uri = f"models:/{MODEL_NAME}/{prod_version}"
            prod_model = mlflow.catboost.load_model(prod_model_uri)
            production_metrics = calculate_metrics(prod_model, x_test, y_test)
    except Exception as e:
        print(f"Could not evaluate production model: {e}")

   # decision logic based on metrics comparison and thresholds
    passed = True
    reasons = []

    if candidate_metrics["roc_auc"] < min_roc_auc:
        passed = False
        reasons.append(
            f"Candidate ROC AUC {candidate_metrics['roc_auc']:.4f} below minimum threshold {min_roc_auc:.4f}"
        )

    if candidate_metrics["roc_auc"] <= production_metrics["roc_auc"]:
        passed = False
        reasons.append(
            f"Candidate ROC AUC {candidate_metrics['roc_auc']:.4f} is not better than production ROC AUC {production_metrics['roc_auc']:.4f}"
        )

    if candidate_metrics["recall"] < production_metrics["recall"] - max_recall_drop:
        passed = False
        reasons.append(
            f"Candidate recall {candidate_metrics['recall']:.4f} dropped too much from production recall {production_metrics['recall']:.4f}"
        )

    result = {"passed": passed,"candidate_run_id": candidate_run_id,"candidate_metrics": candidate_metrics,
              "production_metrics": production_metrics,"reasons": reasons if reasons else ["Candidate passed evaluation"],}

    print("Evaluation result:")
    print(result)

    return result

