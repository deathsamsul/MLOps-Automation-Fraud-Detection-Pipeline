import pandas as pd
from catboost import CatBoostClassifier
import mlflow
import mlflow.catboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from task_2_credit_card_fraud_detecation.features.data_processing import load_and_preprocess_data
from task_2_credit_card_fraud_detecation.utils.utility import CATEGORICAL_COLS, EXPERIMENT_NAME,TRAIN_DATA_PATH





# python -m task_2_credit_card_fraud_detecation.training.train_model

mlflow.set_experiment(EXPERIMENT_NAME)


# now for baseline model training we use just old training data 
# (old+new data) improve later
def train():
    x, y = load_and_preprocess_data(TRAIN_DATA_PATH)

    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )


    with mlflow.start_run() as run:

        params = {"iterations": 1000,"learning_rate": 0.05,"depth": 6,"loss_function": "Logloss", "eval_metric": "AUC",
                   "random_seed": 42}

        mlflow.log_params(params)

        # train model with categorical features
        model = CatBoostClassifier(**params, verbose=0)
        model.fit( x_train, y_train,cat_features=CATEGORICAL_COLS,eval_set=(x_test, y_test),use_best_model=True,
                  verbose_eval=False )


        # evaluate
        pred_proba = model.predict_proba(x_test)[:, 1]
        roc_auc = roc_auc_score(y_test, pred_proba)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model to MLflow registry
        mlflow.catboost.log_model(model,"model",registered_model_name="fraud_detection_model")

        print(f"Run ID: {run.info.run_id}, ROC-AUC: {roc_auc:.4f}")
        return run.info.run_id
    


if __name__ == "__main__":
    train()