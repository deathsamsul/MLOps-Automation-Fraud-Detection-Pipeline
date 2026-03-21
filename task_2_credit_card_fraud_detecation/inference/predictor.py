import pandas as pd
from task_2_credit_card_fraud_detecation.features.data_processing import feature_engineering
from task_2_credit_card_fraud_detecation.mlops.mlflow_utils import load_production_model




_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_production_model()
    return _model


def predict_fraud(input_data: dict):
    
    model = get_model()
    df = pd.DataFrame([input_data])
    df = feature_engineering(df)
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return int(pred), float(prob)