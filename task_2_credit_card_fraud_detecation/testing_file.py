from task_2_credit_card_fraud_detecation.training.evaluate_model import evaluate_candidate_model
from task_2_credit_card_fraud_detecation.mlops.mlflow_utils import register_candidate_model



# python -m py_compile task_2_credit_card_fraud_detecation/airflow_dags/retrain_dag.py
# python -m py_compile task_2_credit_card_fraud_detecation/monitoring/monitoring.py
# python -m py_compile task_2_credit_card_fraud_detecation/pipelines/retrain_pipeline.py
# python -m py_compile task_2_credit_card_fraud_detecation/training/train_model.py
# python -m py_compile task_2_credit_card_fraud_detecation/training/evaluate_model.py
# python -m py_compile task_2_credit_card_fraud_detecation/mlops/mlflow_utils.py

# -------liberary imports could be tested by running the module directly

# python -m task_2_credit_card_fraud_detecation.monitoring.monitoring
# python -m task_2_credit_card_fraud_detecation.training.train_model
# python -m task_2_credit_card_fraud_detecation.pipelines.retrain_pipeline



# python -m task_2_credit_card_fraud_detecation.training.train_model
# 2026/03/20 18:45:35 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
# Run ID: 2877235ba863465198170de3abb5ee76, ROC-AUC: 0.9981
# {'run_id': '2877235ba863465198170de3abb5ee76', 'roc_auc': 0.9980854435308042}


run_id = '2877235ba863465198170de3abb5ee76'



print(evaluate_candidate_model(candidate_run_id=run_id))
print(register_candidate_model(candidate_run_id=run_id))

# task_2_credit_card_fraud_detecation/testing_file.py