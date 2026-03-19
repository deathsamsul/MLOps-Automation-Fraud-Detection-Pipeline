from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator  
from datetime import datetime, timedelta
from task_2_credit_card_fraud_detecation.pipelines.retrain_pipeline import run_retraining_pipeline
import logging
from airflow.exceptions import AirflowFailException, AirflowSkipException

from task_2_credit_card_fraud_detecation.training.evaluate_model import evaluate_candidate_model
from task_2_credit_card_fraud_detecation.mlops.mlflow_utils import register_candidate_model


default_args = {"owner": "mlops","depends_on_past": False,"email_on_failure": True,"email_on_retry": False,
                "retries": 2,"retry_delay": timedelta(minutes=5),}


def retrain_wrapper(**context):
    
    logging.info("Starting retraining pipeline...")
    result = run_retraining_pipeline(force=True)  # force=True that checck_performance_drop and run_drift_detection will be skipped 

    if result is None:
        raise AirflowFailException("Retraining pipeline returned None")
    logging.info("Retraining completed: %s", result)
    return result


def evaluate_wrapper(**context):
    """
    Evaluate the candidate model.
    Expected behavior:
    - compare candidate vs threshold or production model
    - return True or metadata if passed
    """
    logging.info("Evaluating candidate model...")

    result = evaluate_candidate_model()

    if not result:
        raise AirflowFailException("Candidate model failed evaluation checks")

    logging.info("Candidate model passed evaluation")
    return result


def register_wrapper(**context):
    """
    Register the validated candidate model in MLflow.
    """
    logging.info("Registering candidate model...")

    result = register_candidate_model()

    if result is None:
        raise AirflowFailException("Model registration failed")

    logging.info("Model registration completed: %s", result)
    return result


with DAG(
    dag_id="fraud_retrain_4step_pipeline",
    default_args=default_args,
    description="4-step fraud detection retraining DAG: monitor -> retrain -> evaluate -> register",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=2),
    tags=["fraud", "mlops", "retraining", "airflow"],
) as dag:

    monitor_task = PythonOperator(
        task_id="monitor_model",
        python_callable=monitor_wrapper,
        execution_timeout=timedelta(minutes=20),
    )

    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_wrapper,
        execution_timeout=timedelta(hours=1),
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_wrapper,
        execution_timeout=timedelta(minutes=30),
    )

    register_task = PythonOperator(
        task_id="register_model",
        python_callable=register_wrapper,
        execution_timeout=timedelta(minutes=15),
    )

    monitor_task >> retrain_task >> evaluate_task >> register_task