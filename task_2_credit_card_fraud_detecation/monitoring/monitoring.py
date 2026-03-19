import sqlite3
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import mlflow
from task_2_credit_card_fraud_detecation.utils.utility import  TRAIN_DATA_PATH,DB_PATH, CSV_PATH
import json
import os
from sklearn.metrics import f1_score



mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud_monitoring")  



def check_performance_drop(threshold_f1=0.80, use_csv=False):

    try:
        if use_csv:
            df = pd.read_csv(CSV_PATH)
            df = df[df['actual_label'].notna()]

        else:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT * FROM predictions WHERE actual_label IS NOT NULL", conn)
            conn.close()

        if len(df) < 10:
            print(f"Not enough labeled data ({len(df)} < 10). Skipping performance check.")
            return False

        y_true = df['actual_label'].astype(int)
        y_pred = df['prediction'].astype(int)

        f1 = f1_score(y_true, y_pred)
        print(f"Current F1 Score: {f1:.4f}")

        with mlflow.start_run(run_name="performance_check"):
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("labeled_samples", len(df))

        if f1 < threshold_f1:
            print("Performance dropped below threshold. Retraining needed.")
            return True
        return False
    except Exception as e:
        print(f"Error during performance check: {e}")
        return False
    





def run_drift_detection(reference_path=TRAIN_DATA_PATH, current_path=CSV_PATH,drift_threshold=0.30):

    # select only the feature columns
    feature_cols = ['merchant', 'category', 'amt', 'gender', 'city','state', 'zip', 'lat', 'long',
                    'city_pop', 'job', 'unix_time','merch_lat', 'merch_long', 'trans_date_trans_time', 'dob']
    
    try:
        reference1 = pd.read_csv(reference_path).sample(5000)
        current1 = pd.read_csv(current_path)
        if reference1.empty:
                print("Reference data is empty. Skipping drift detection.")
                return False

        if current1.empty:
                print("Current data is empty. Skipping drift detection.")
                return False

        reference = reference1[feature_cols]
        current = current1[feature_cols]

        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=reference, current_data=current)


        drift_summary = json.loads(result.json())     #Converts JSON string → Python dictionary
        metric_data = drift_summary["metrics"][0]["value"]
        n_drifted = metric_data["count"]   # number of columns with detected drift 
        drift_ratio = metric_data["share"] # ratio of columns with detected drift out of total columns analyzed
        total = len(feature_cols)


        print(f"Drift detected in {n_drifted}/{total} columns ({drift_ratio:.2%})")

        report_path = os.path.join(os.path.dirname(__file__), "drift_report.html")
        result.save_html(report_path)

        
        with mlflow.start_run(run_name="drift_check"):
            mlflow.log_metric("drifted_columns", n_drifted)
            mlflow.log_metric("drift_ratio", drift_ratio)
            mlflow.log_artifact(report_path)


        return drift_ratio > drift_threshold
    except Exception as e:
        print(f"Error during drift detection: {e}")
        return False     


def monitoring_pipeline():  #        False - system healthy / no retraining needed

    performance_drop = check_performance_drop()
    data_drift = run_drift_detection()

    if performance_drop or data_drift:
        print(" Trigger retraining pipeline")
        return True
    else:
        print("System healthy")
        return False
    

if __name__ == "__main__":
    result = monitoring_pipeline()
    print("Monitoring pipeline result:", result)