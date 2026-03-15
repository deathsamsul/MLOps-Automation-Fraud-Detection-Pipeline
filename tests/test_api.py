import pytest
from fastapi.testclient import TestClient
from task_2_credit_card_fraud_detecation.api.api import app

client = TestClient(app)

def test_predict():
    sample = {
        "merchant": "Amazon",
        "category": "shopping_net",
        "amt": 45.20,
        "gender": "F",
        "city": "New York",
        "state": "NY",
        "zip": 10001,
        "lat": 40.7128,
        "long": -74.0060,
        "city_pop": 8000000,
        "job": "Engineer",
        "unix_time": 1371816917,
        "merch_lat": 40.7306,
        "merch_long": -73.9352,
        "trans_date_trans_time": "2023-01-01 14:30:00",
        "dob": "1990-01-01"
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "prediction" in data
    assert 0 <= data["fraud_probability"] <= 1