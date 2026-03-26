FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY task_2_credit_card_fraud_detecation /app/task_2_credit_card_fraud_detecation

ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data
ENV DB_DIR=/app/database
ENV MLFLOW_DIR=/app/mlruns
ENV DB_PATH=/app/database/fraud_monitor.db
ENV CSV_PATH=/app/data/predictions.csv
ENV TRAIN_DATA_PATH=/app/data/fraud_train.csv
ENV TEST_DATA_PATH=/app/data/fraud_test.csv
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

EXPOSE 8501

CMD ["streamlit", "run", "task_2_credit_card_fraud_detecation/monitoring/monitoring_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]