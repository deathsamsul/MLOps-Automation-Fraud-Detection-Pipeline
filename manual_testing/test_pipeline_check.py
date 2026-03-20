from task_2_credit_card_fraud_detecation.pipelines.retrain_pipeline import run_retraining_pipeline
from task_2_credit_card_fraud_detecation.training.evaluate_model import evaluate_candidate_model
from task_2_credit_card_fraud_detecation.mlops.mlflow_utils import register_candidate_model



# python -m manual_testing.test_pipeline_check

result = run_retraining_pipeline()
print("TRAIN RESULT:", result)

eval_result = evaluate_candidate_model(candidate_run_id=result["run_id"])
print("EVAL RESULT:", eval_result)

if eval_result["passed"]:
    reg_result = register_candidate_model(candidate_run_id=result["run_id"])
    print("REGISTER RESULT:", reg_result)
else:
    print("Model did not pass evaluation")