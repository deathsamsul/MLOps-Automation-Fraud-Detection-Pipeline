import shutil
from task_2_credit_card_fraud_detecation.utils.utility import TRAIN_DATA_PATH, REFERENCE_DATA_PATH


# python -m task_2_credit_card_fraud_detecation.scripts.create_reference


# Copy training data to reference_data.csv
shutil.copyfile(TRAIN_DATA_PATH, REFERENCE_DATA_PATH)
print(f"Copied {TRAIN_DATA_PATH} to {REFERENCE_DATA_PATH}")