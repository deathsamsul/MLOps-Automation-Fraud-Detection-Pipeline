FROM apache/airflow:3.1.8-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

WORKDIR /opt/airflow

COPY airflow_requirements.txt /opt/airflow/airflow_requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /opt/airflow/airflow_requirements.txt

COPY task_2_credit_card_fraud_detecation /opt/airflow/task_2_credit_card_fraud_detecation
COPY docker/fraud-airflow/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENV PYTHONPATH=/opt/airflow
ENV DATA_DIR=/opt/airflow/data
ENV DB_DIR=/opt/airflow/database
ENV MLFLOW_DIR=/opt/airflow/mlruns
ENV DB_PATH=/opt/airflow/database/fraud_monitor.db
ENV CSV_PATH=/opt/airflow/data/predictions.csv
ENV TRAIN_DATA_PATH=/opt/airflow/data/fraud_train.csv
ENV TEST_DATA_PATH=/opt/airflow/data/fraud_test.csv
ENV MLFLOW_TRACKING_URI=file:/opt/airflow/mlruns
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/task_2_credit_card_fraud_detecation/airflow_dags
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////opt/airflow/airflow.db