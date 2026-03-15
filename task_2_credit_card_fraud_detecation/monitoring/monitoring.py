import sqlite3
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import mlflow
from task_2_credit_card_fraud_detecation.utils.utility import DB_PATH, REFERENCE_DATA_PATH, CSV_PATH





def check_performance_drop(threshold_f1=0.80, use_csv=False):

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

    from sklearn.metrics import f1_score
    f1 = f1_score(y_true, y_pred)
    print(f"Current F1 Score: {f1:.4f}")

    if f1 < threshold_f1:
        print("Performance dropped below threshold. Retraining needed.")
        return True
    return False





def run_drift_detection(reference_path=REFERENCE_DATA_PATH, current_path=CSV_PATH):

    # Select only the feature columns
    feature_cols = ['merchant', 'category', 'amt', 'gender', 'city','state', 'zip', 'lat', 'long',
                    'city_pop', 'job', 'unix_time','merch_lat', 'merch_long', 'trans_date_trans_time', 'dob']
    

    reference1 = pd.read_csv(reference_path)
    current1 = pd.read_csv(current_path)

    reference = reference1[feature_cols]
    current = current1[feature_cols]

    # create evidently report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    # Extract drift summary
    drift_summary = report.as_dict()
    n_drifted = drift_summary['metrics'][0]['result']['number_of_drifted_columns']
    total = drift_summary['metrics'][0]['result']['number_of_columns']
    drift_ratio = n_drifted / total if total > 0 else 0

    # save report locally
    report.save_html("drift_report.html")
    print(f"Drift detected in {n_drifted}/{total} columns ({drift_ratio:.2%})")

    # log metrics and report to MLflow
    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_metric("drifted_columns", n_drifted)
        mlflow.log_metric("drift_ratio", drift_ratio)
        mlflow.log_artifact("drift_report.html")

    # Trigger alert if more than 10% of features drifted
    return drift_ratio > 0.1




def monitoring_pipeline():

    performance_drop = check_performance_drop()
    data_drift = run_drift_detection()

    if performance_drop or data_drift:
        print(" Trigger retraining pipeline")
        return True
    else:
        print("System healthy")
        return False