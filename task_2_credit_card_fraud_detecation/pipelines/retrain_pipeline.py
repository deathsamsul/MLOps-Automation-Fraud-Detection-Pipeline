import logging
from task_2_credit_card_fraud_detecation.training.train_model import train




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def run_retraining_pipeline():
    
    logger.info("Starting retraining pipeline...")
    train_result = train()

    if not train_result:
        raise ValueError("Training failed: no result returned")

    logger.info("Retraining completed successfully: %s", train_result)
    return train_result


if __name__ == "__main__":
    result = run_retraining_pipeline()
    print(result)
