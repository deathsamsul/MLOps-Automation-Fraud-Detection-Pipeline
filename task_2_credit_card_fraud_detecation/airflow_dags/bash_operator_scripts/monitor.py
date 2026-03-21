import json
from task_2_credit_card_fraud_detecation.monitoring.monitoring import run_monitoring_pipeline


def main():
    result = run_monitoring_pipeline()
    if not result:
        raise Exception("No retraining needed")

    print(json.dumps({"should_retrain": True}))


if __name__ == "__main__":
    main()