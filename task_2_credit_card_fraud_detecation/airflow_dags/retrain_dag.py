from datetime import datetime, timedelta
import logging
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.exceptions import AirflowFailException, AirflowSkipException
from task_2_credit_card_fraud_detecation.monitoring.monitoring import run_monitoring_pipeline
from task_2_credit_card_fraud_detecation.pipelines.retrain_pipeline import run_retraining_pipeline
from task_2_credit_card_fraud_detecation.training.evaluate_model import evaluate_candidate_model
from task_2_credit_card_fraud_detecation.mlops.mlflow_utils import register_candidate_model





default_args = {"owner": "autometed_mlops","depends_on_past": False,"email_on_failure": True,
                "email_on_retry": False,"retries": 2,"retry_delay": timedelta(minutes=5),}


def monitor_wrapper(**context):
    logging.info("Running monitoring checks...")
    should_retrain = run_monitoring_pipeline()

    if not should_retrain:
        raise AirflowSkipException("No drift/performance issue detected. Skipping DAG.")

    logging.info("Monitoring detected issue. Continue to retraining.")
    return {"should_retrain": True}


def retrain_wrapper(**context):
    logging.info("Starting retraining pipeline...")
    result = run_retraining_pipeline()   # force=True that checck_performance_drop and run_drift_detection will be skipped 

    if result is None:
        raise AirflowFailException("Retraining pipeline returned None")

    if "run_id" not in result:
        raise AirflowFailException("Retraining pipeline did not return run_id")

    logging.info("Retraining completed: %s", result)
    return result


def evaluate_wrapper(**context):
    logging.info("Evaluating candidate model...")

    ti = context["ti"]
    retrain_result = ti.xcom_pull(task_ids="retrain_model")
    if not retrain_result:
        raise AirflowFailException("No retrain result found in XCom")

    candidate_run_id = retrain_result["run_id"]
    result = evaluate_candidate_model(candidate_run_id=candidate_run_id)
    if not result.get("passed", False):
        raise AirflowFailException(
            f"Candidate model failed evaluation checks: {result.get('reasons', [])}"
        )

    logging.info("Candidate model passed evaluation: %s", result)
    return result


def register_wrapper(**context):
    logging.info("Registering validated candidate model...")

    ti = context["ti"]
    eval_result = ti.xcom_pull(task_ids="evaluate_model")
    if not eval_result:
        raise AirflowFailException("No evaluation result found in XCom")

    candidate_run_id = eval_result["candidate_run_id"]
    result = register_candidate_model(candidate_run_id=candidate_run_id)
    if result is None:
        raise AirflowFailException("Model registration failed")

    logging.info("Model registration completed: %s", result)
    return result


with DAG( dag_id="fraud_retrain_4step_pipeline",
         default_args=default_args,
         description="4-step fraud detection pipeline: monitor -> retrain -> evaluate -> register",
         start_date=datetime(2026, 2, 1),
         schedule="@weekly",
         catchup=False,
         max_active_runs=1,
         dagrun_timeout=timedelta(hours=2),
         tags=["fraud", "mlops", "retraining", "airflow"], ) as dag:

    monitor_task = PythonOperator(  task_id="monitor_model",python_callable=monitor_wrapper,
                                  execution_timeout=timedelta(minutes=20),)

    retrain_task = PythonOperator( task_id="retrain_model",python_callable=retrain_wrapper,
                                  execution_timeout=timedelta(hours=1),)

    evaluate_task = PythonOperator(task_id="evaluate_model",python_callable=evaluate_wrapper,
                                   execution_timeout=timedelta(minutes=30),)

    register_task = PythonOperator( task_id="register_model",python_callable=register_wrapper,
                                   execution_timeout=timedelta(minutes=15),)

    monitor_task >> retrain_task >> evaluate_task >> register_task




