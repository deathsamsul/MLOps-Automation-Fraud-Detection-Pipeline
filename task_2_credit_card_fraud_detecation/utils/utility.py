import sqlite3
import os
import pandas as pd
from contextlib import contextmanager









MODEL_COLUMNS = [
    'merchant', 'category', 'amt', 'gender', 'city', 'state', 'zip', 'lat',
    'long', 'city_pop', 'job', 'unix_time', 'merch_lat', 'merch_long',
    'hour', 'day', 'month', 'weekday', 'age', 'distance', 'amt_log',
    'is_night', 'is_weekend'
]

CATEGORICAL_COLS = ['merchant', 'category', 'gender', 'city', 'state', 'job']

# Docker-friendly base paths
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
DB_DIR = os.environ.get("DB_DIR", "/app/database")
MLFLOW_DIR = os.environ.get("MLFLOW_DIR", "/app/mlruns")

DB_PATH = os.environ.get("DB_PATH", os.path.join(DB_DIR, "fraud_monitor.db"))
CSV_PATH = os.environ.get("CSV_PATH", os.path.join(DATA_DIR, "predictions.csv"))
TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH", os.path.join(DATA_DIR, "fraud_train.csv"))
TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH", os.path.join(DATA_DIR, "fraud_test.csv"))

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"file:{MLFLOW_DIR}")

MODEL_NAME = "fraud_detection_model"
EXPERIMENT_NAME = "fraud_detection"

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
os.makedirs(MLFLOW_DIR, exist_ok=True)

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                transaction_id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                fraud_probability REAL,
                prediction INTEGER,
                actual_label INTEGER
            )
        """)
        conn.commit()

def init_csv():
    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(columns=[
            'transaction_id', 'timestamp', 'fraud_probability', 'prediction', 'actual_label',
            'merchant', 'category', 'amt', 'gender', 'city', 'state', 'zip', 'lat', 'long',
            'city_pop', 'job', 'unix_time', 'merch_lat', 'merch_long',
            'trans_date_trans_time', 'dob'
        ])
        df.to_csv(CSV_PATH, index=False)

def append_prediction_to_csv(record: dict):
    df = pd.DataFrame([record])
    df.to_csv(CSV_PATH, mode='a', header=not os.path.exists(CSV_PATH), index=False)

def update_label_in_csv(transaction_id: str, actual_label: int):
    df = pd.read_csv(CSV_PATH)
    if transaction_id not in df['transaction_id'].values:
        raise ValueError(f"Transaction ID {transaction_id} not found in CSV")
    df.loc[df['transaction_id'] == transaction_id, 'actual_label'] = actual_label
    df.to_csv(CSV_PATH, index=False)

def load_predictions_from_csv():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return pd.DataFrame()














# import sqlite3
# import os
# import pandas as pd
# from contextlib import contextmanager






# MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"



# MODEL_COLUMNS = [
#     'merchant', 'category', 'amt', 'gender', 'city', 'state', 'zip', 'lat',
#     'long', 'city_pop', 'job', 'unix_time', 'merch_lat', 'merch_long',
#     'hour', 'day', 'month', 'weekday', 'age', 'distance', 'amt_log',
#     'is_night', 'is_weekend'
# ]

# CATEGORICAL_COLS = ['merchant', 'category', 'gender', 'city', 'state', 'job']

# #  configuration
# DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'database', 'fraud_monitor.db')   # store predictions and labels
# CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions.csv')     # store full prediction records for monitoring and drift detection
# TRAIN_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'fraud_train.csv') # trainign data
# TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'fraud_test.csv') # testing data

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # project root
# MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'file:' + os.path.join(BASE_DIR, 'mlruns')) # local mlflow tracking URI

# MODEL_NAME = "fraud_detection_model"
# EXPERIMENT_NAME = "fraud_detection"

# os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
# os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)




# #database helpers 
# @contextmanager
# def get_db_connection():
#     conn = sqlite3.connect(DB_PATH)
#     try:
#         yield conn
#     finally:
#         conn.close()

# def init_db():

#     with get_db_connection() as conn:
#         conn.execute("""CREATE TABLE IF NOT EXISTS predictions (transaction_id TEXT PRIMARY KEY,
#                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
#                     fraud_probability REAL,prediction INTEGER,actual_label INTEGER)
#                      """)
        
#         conn.commit()


# #CSV helpers 
# def init_csv():
#     if not os.path.exists(CSV_PATH):

#         df = pd.DataFrame(columns=['transaction_id', 'timestamp', 'fraud_probability', 'prediction','actual_label',
#                                    'merchant', 'category', 'amt', 'gender', 'city','state', 'zip', 'lat', 'long', 'city_pop',
#                                      'job', 'unix_time','merch_lat', 'merch_long', 'trans_date_trans_time', 'dob' ])
        
#         df.to_csv(CSV_PATH, index=False)



# def append_prediction_to_csv(record: dict):
#     df = pd.DataFrame([record])
#     df.to_csv(CSV_PATH, mode='a', header=not os.path.exists(CSV_PATH), index=False)




# def update_label_in_csv(transaction_id: str, actual_label: int):
#     df = pd.read_csv(CSV_PATH)
#     if transaction_id not in df['transaction_id'].values:
#         raise ValueError(f"Transaction ID {transaction_id} not found in CSV")
    
#     df.loc[df['transaction_id'] == transaction_id, 'actual_label'] = actual_label
#     df.to_csv(CSV_PATH, index=False)


# def load_predictions_from_csv():
#     if os.path.exists(CSV_PATH):
#         return pd.read_csv(CSV_PATH)
#     return pd.DataFrame()



