import json
from task_2_credit_card_fraud_detecation.pipelines.retrain_pipeline import run_retraining_pipeline



# python -m task_2_credit_card_fraud_detecation.bash_operator_scripts.retrain


def main():
    result = run_retraining_pipeline()

    if result is None or "run_id" not in result:
        raise Exception("Invalid retrain result")

    print(json.dumps(result))


if __name__ == "__main__":
    main()