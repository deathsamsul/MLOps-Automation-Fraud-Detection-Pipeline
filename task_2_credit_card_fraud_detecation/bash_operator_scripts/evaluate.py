import json
import sys
from task_2_credit_card_fraud_detecation.training.evaluate_model import evaluate_candidate_model

# python -m task_2_credit_card_fraud_detecation.bash_operator_scripts.evaluate 93c4afbcce784984ae9ec98fa627be90

def main(run_id):

    result = evaluate_candidate_model(candidate_run_id=run_id)
    if not result.get("passed", False):
        raise Exception(f"Evaluation failed: {result.get('reasons', [])}")

    print(json.dumps(result))



if __name__ == "__main__":
    run_id = sys.argv[1]
    main(run_id)