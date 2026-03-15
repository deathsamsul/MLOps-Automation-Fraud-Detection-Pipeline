import pandas as pd
import numpy as np
from task_2_credit_card_fraud_detecation.utils.utility import MODEL_COLUMNS




def load_and_preprocess_data(file_path: str, target_col: str = 'is_fraud'):


    df = pd.read_csv(file_path)
    y = df[target_col] if target_col in df.columns else None
    X = df.drop(columns=[target_col], errors='ignore')
    X = feature_engineering(X)
    return X, y




def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:


    df = df.copy()

    # convert datetime columns
    if 'trans_date_trans_time' in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    if 'dob' in df.columns:
        df["dob"] = pd.to_datetime(df["dob"])


    # Time features
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["weekday"] = df["trans_date_trans_time"].dt.weekday

    # age
    df["age"] = df["trans_date_trans_time"].dt.year - df["dob"].dt.year

    # distance between cardholder and merchant
    df["distance"] = np.sqrt((df["lat"] - df["merch_lat"])**2 + (df["long"] - df["merch_long"])**2)

    # log of transaction amount
    df["amt_log"] = np.log1p(df["amt"])

    # night transaction (22:00 - 04:00)
    df["is_night"] = df["hour"].apply(lambda x: 1 if (x >= 22 or x <= 4) else 0)

    # weekend transaction (weekday >=5)
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)

    # drop raw/unused columns
    drop_cols = ["Unnamed: 0", "first", "last", "street", "cc_num", "trans_num","trans_date_trans_time", "dob"]
    
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # ensure only MODEL_COLUMNS are present and in correct order
    df = df[MODEL_COLUMNS]
    
    return df