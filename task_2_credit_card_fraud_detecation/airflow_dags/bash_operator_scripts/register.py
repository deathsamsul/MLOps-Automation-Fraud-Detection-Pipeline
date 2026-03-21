import json
import sys
from task_2_credit_card_fraud_detecation.mlops.mlflow_utils import register_candidate_model




def main(run_id):
    
    result = register_candidate_model(candidate_run_id=run_id)

    if result is None:
        raise Exception("Registration failed")

    print(json.dumps(result))



if __name__ == "__main__":
    run_id = sys.argv[1]
    main(run_id)