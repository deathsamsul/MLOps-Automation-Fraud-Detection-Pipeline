from task_2_credit_card_fraud_detecation.utils.utility import init_db, get_db_connection
import os

def test_db_init():
    init_db()
    assert os.path.exists('database/fraud_monitor.db')