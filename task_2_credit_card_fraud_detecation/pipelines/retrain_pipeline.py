import mlflow
from task_2_credit_card_fraud_detecation.training.train_model import train
from task_2_credit_card_fraud_detecation.mlops.promote_model import promote_to_production, archive_current_production
from task_2_credit_card_fraud_detecation.mlops.mlflow_utils import log_model_to_registry
from task_2_credit_card_fraud_detecation.monitoring.monitoring import check_performance_drop, run_drift_detection
import logging




# task_2_credit_card_fraud_detecation/pipelines/retrain_pipeline.py


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_retraining_pipeline(force=False):
    """
    Full retraining pipeline:
    1. Check performance or drift (skip if force=True)
    2. Train new model
    3. Evaluate and compare with current production
    4. If better, promote to production
    """

    if not force:
        perf_drop = check_performance_drop()
        drift = run_drift_detection()
        if not (perf_drop or drift):
            logger.info("No performance drop or drift detected. Skipping retraining.")
            return

    logger.info("Starting retraining pipeline...")


    # Train new model
    new_run_id = train()

    # Load current production model's metrics for comparison
    client = mlflow.MlflowClient()

    try:
        prod_versions = client.get_latest_versions("fraud_detection_model", stages=["Production"])
        if prod_versions:
            prod_run_id = prod_versions[0].run_id
            prod_metrics = client.get_run(prod_run_id).data.metrics
            prod_roc = prod_metrics.get("roc_auc", 0)
        else:
            prod_roc = 0
    except:
        prod_roc = 0

    # Get new model's metrics
    new_run = client.get_run(new_run_id)
    new_roc = new_run.data.metrics.get("roc_auc", 0)

    logger.info(f"Production ROC-AUC: {prod_roc:.4f}, New ROC-AUC: {new_roc:.4f}")


    if new_roc > prod_roc:
        # Archive current production and promote new version
        archive_current_production()
        # The new model is already registered via train(), get its version
        versions = client.get_latest_versions("fraud_detection_model", stages=["None"])
        new_version = versions[0].version if versions else None
        
        if new_version:
            promote_to_production(new_version)
            logger.info(f"New model version {new_version} promoted to Production.")
        else:
            logger.error("Could not find new model version.")
    else:
        logger.info("New model did not outperform production. Keeping current model.")

if __name__ == "__main__":
    run_retraining_pipeline(force=True)  # Set force=False to use checks