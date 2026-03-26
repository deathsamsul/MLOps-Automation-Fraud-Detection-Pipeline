import json
import sys
from task_2_credit_card_fraud_detecation.monitoring.monitoring import run_monitoring_pipeline




def main():
    result = run_monitoring_pipeline()

    if not result:
        print(json.dumps({"should_retrain": False, "message": "No retraining needed"}))
        sys.exit(99)   # tell Airflow to mark task as skipped

    print(json.dumps({"should_retrain": True}))

if __name__ == "__main__":
    main()

# import json
# import os
# import sys
# from task_2_credit_card_fraud_detecation.monitoring.monitoring import run_monitoring_pipeline

# def main():
#     force_retrain = os.getenv("FORCE_RETRAIN", "false").lower() == "true"

#     if force_retrain:
#         print(json.dumps({"should_retrain": True, "message": "Forced retrain for testing"}))
#         sys.exit(0)

#     result = run_monitoring_pipeline()

#     if not result:
#         print(json.dumps({"should_retrain": False, "message": "No retraining needed"}))
#         sys.exit(99)

#     print(json.dumps({"should_retrain": True}))

# if __name__ == "__main__":
#     main()