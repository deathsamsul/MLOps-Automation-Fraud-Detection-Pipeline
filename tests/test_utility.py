import os
import pytest
import pandas as pd
from task_2_credit_card_fraud_detecation.utils import utility as util

@pytest.fixture(autouse=True)
def override_paths(monkeypatch, tmp_path):
    temp_db = tmp_path / "test_fraud_monitor.db"
    temp_csv = tmp_path / "test_predictions.csv"
    monkeypatch.setattr('task_2_credit_card_fraud_detecation.utils.utility.DB_PATH', str(temp_db))
    monkeypatch.setattr('task_2_credit_card_fraud_detecation.utils.utility.CSV_PATH', str(temp_csv))
    # Ensure parent dirs exist (optional, as tmp_path already exists)
    temp_db.parent.mkdir(parents=True, exist_ok=True)
    temp_csv.parent.mkdir(parents=True, exist_ok=True)

def test_db_init():
    util.init_db()
    assert os.path.exists(util.DB_PATH)

def test_db_insert_and_select():
    util.init_db()
    with util.get_db_connection() as conn:
        conn.execute("INSERT INTO predictions (transaction_id, fraud_probability, prediction) VALUES (?, ?, ?)",
                     ("txn123", 0.75, 1))
        conn.commit()
        cursor = conn.execute("SELECT * FROM predictions WHERE transaction_id = ?", ("txn123",))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "txn123"
        assert row[2] == 0.75
        assert row[3] == 1

def test_csv_init():
    util.init_csv()
    assert os.path.exists(util.CSV_PATH)
    df = pd.read_csv(util.CSV_PATH)
    assert df.empty




def test_append_prediction_to_csv():
    util.init_csv()
    record = {
        'transaction_id': 'txn456',
        'timestamp': '2023-01-01T12:00:00',
        'fraud_probability': 0.9,
        'prediction': 1,
        'actual_label': None,
        'merchant': 'Test',
        'category': 'test',
        'amt': 100.0,
        'gender': 'M',
        'city': 'TestCity',
        'state': 'TS',
        'zip': 12345,
        'lat': 0.0,
        'long': 0.0,
        'city_pop': 1000,
        'job': 'Tester',
        'unix_time': 123456789,
        'merch_lat': 1.0,
        'merch_long': 1.0,
        'trans_date_trans_time': '2023-01-01 12:00:00',
        'dob': '2000-01-01'
    }
    util.append_prediction_to_csv(record)
    df = util.load_predictions_from_csv()
    assert len(df) == 1
    assert df.iloc[0]['transaction_id'] == 'txn456'



def test_update_label_in_csv():
    util.init_csv()
    record = {
        'transaction_id': 'txn789',
        'timestamp': '2023-01-01T12:00:00',
        'fraud_probability': 0.2,
        'prediction': 0,
        'actual_label': None,
        'merchant': 'Test',
        'category': 'test',
        'amt': 10.0,
        'gender': 'F',
        'city': 'TestCity',
        'state': 'TS',
        'zip': 12345,
        'lat': 0.0,
        'long': 0.0,
        'city_pop': 1000,
        'job': 'Tester',
        'unix_time': 123456789,
        'merch_lat': 1.0,
        'merch_long': 1.0,
        'trans_date_trans_time': '2023-01-01 12:00:00',
        'dob': '2000-01-01'
    }
    util.append_prediction_to_csv(record)

    # Verify record exists
    df_before = util.load_predictions_from_csv()
    assert len(df_before) == 1
    assert df_before.iloc[0]['transaction_id'] == 'txn789'
    assert pd.isna(df_before.iloc[0]['actual_label'])

    # Update label
    util.update_label_in_csv('txn789', 1)

    # Verify update
    df_after = util.load_predictions_from_csv()
    assert len(df_after) == 1
    updated_label = df_after.loc[df_after['transaction_id'] == 'txn789', 'actual_label'].iloc[0]
    assert updated_label == 1, f"Expected 1, got {updated_label} (type: {type(updated_label)})"

