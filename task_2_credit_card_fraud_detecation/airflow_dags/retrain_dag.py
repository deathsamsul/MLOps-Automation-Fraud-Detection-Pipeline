from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from task_2_credit_card_fraud_detecation.pipelines.retrain_pipeline import run_retraining_pipeline




default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "fraud_retrain_pipeline",
    default_args=default_args,
    description="Retrain fraud detection model weekly",
    schedule_interval="@weekly",
    catchup=False,
)

retrain_task = PythonOperator(
    task_id="retrain_model",
    python_callable=run_retraining_pipeline,
    op_kwargs={"force": False},  # Use drift/performance checks
    dag=dag,
)