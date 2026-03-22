from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator



default_args = {
    "owner": "automated_mlops",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


#python -m task_2_credit_card_fraud_detecation.airflow_dags.retrain_dag
# export AIRFLOW__CORE__DAGS_FOLDER=/mnt/c/Users/samsu/Desktop/code/fraud_detection/task_2_credit_card_fraud_detecation/airflow_dags
# task_2_credit_card_fraud_detecation/airflow_dags/retrain_dag.py

# source /mnt/c/Users/samsu/Desktop/code/fraud_detection/ml_env/bin/activate
# python -m task_2_credit_card_fraud_detecation.airflow_dags.bash_operator_scripts.monitor

ML_ENV = "/mnt/c/Users/samsu/Desktop/code/fraud_detection/ml_env/bin/activate"
PROJECT_DIR = "/mnt/c/Users/samsu/Desktop/code/fraud_detection"

SCRIPTS_DIR = f"{PROJECT_DIR}/task_2_credit_card_fraud_detecation/airflow_dags/bash_operator_scripts"



with DAG(
    dag_id="fraud_retrain_4step_pipeline",
    default_args=default_args,
    start_date=datetime(2026, 2, 1),
    schedule="@weekly",
    catchup=False,
) as dag:

    monitor_task = BashOperator(
        task_id="monitor_model",
        bash_command=f"""
        set -e
        source {ML_ENV}
        cd {PROJECT_DIR}
        python {SCRIPTS_DIR}/monitor.py
        """,
    )

    retrain_task = BashOperator(
        task_id="retrain_model",
        bash_command=f"""
        set -e
        source {ML_ENV}
        cd {PROJECT_DIR}
        python {SCRIPTS_DIR}/retrain.py > /tmp/retrain_output.json
        """,
    )

    evaluate_task = BashOperator(
        task_id="evaluate_model",
        bash_command=f"""
        set -e
        source {ML_ENV}
        cd {PROJECT_DIR}

        RUN_ID=$(cat /tmp/retrain_output.json | jq -r '.run_id')
        python {SCRIPTS_DIR}/evaluate.py $RUN_ID > /tmp/eval_output.json
        """,
    )

    register_task = BashOperator(
        task_id="register_model",
        bash_command=f"""
        set -e
        source {ML_ENV}
        cd {PROJECT_DIR}

        RUN_ID=$(cat /tmp/eval_output.json | jq -r '.candidate_run_id')
        python {SCRIPTS_DIR}/register.py $RUN_ID
        """,
    )

    monitor_task >> retrain_task >> evaluate_task >> register_task