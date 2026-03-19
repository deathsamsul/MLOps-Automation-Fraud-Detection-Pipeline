from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import uuid
from datetime import datetime
from task_2_credit_card_fraud_detecation.inference.predictor import predict_fraud
from task_2_credit_card_fraud_detecation.utils.utility import (init_db, get_db_connection, init_csv,append_prediction_to_csv,
                                                                update_label_in_csv)





# uvicorn task_2_credit_card_fraud_detecation.api.api:app --reload

app = FastAPI(title="Fraud Detection API")

# initialize storage
init_db()
init_csv()


# input data validation 
class Transaction(BaseModel):
    merchant: str = "fraud_Rippin, Kub and Mann"
    category: str = "misc_net"
    amt: float = 100.0
    gender: str = "M"
    city: str = "Altona"
    state: str = "NY"
    zip: int = 12910
    lat: float = 44.8865
    long: float = -73.5766
    city_pop: int = 4304
    job: str = "Furniture designer"
    unix_time: int = 1371816865
    merch_lat: float = 44.959148
    merch_long: float = -73.6224
    trans_date_trans_time: str = "2013-06-21 12:14:25"
    dob: str = "1968-03-19"


# label update validation
class LabelUpdate(BaseModel):
    transaction_id: str
    actual_label: int= Field(..., ge=0, le=1, description="Actual label must be 0 (legit) or 1 (fraud)")



# predict endpoint
@app.post("/predict")
def predict(transaction: Transaction):

    
    input_dict = transaction.model_dump() 
    pred, prob = predict_fraud(input_dict)


    # generate transaction ID and timestamp
    transaction_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()


    # prepare record for CSV (include all input fields)
    record = {'transaction_id': transaction_id,'timestamp': timestamp,'fraud_probability': prob,'prediction': pred,
              'actual_label': None, **input_dict            # merge all transaction fields
              }


    # store in SQLite (only metadata)
    with get_db_connection() as conn:
        conn.execute("""INSERT INTO predictions (transaction_id, timestamp, fraud_probability, prediction) VALUES (?, ?, ?, ?)""",
                      (transaction_id, timestamp, prob, pred))
        conn.commit()


    # append to CSV (full record)
    append_prediction_to_csv(record)

    return {"transaction_id": transaction_id,"fraud_probability": prob,"prediction": pred}






@app.post("/update_label")
def update_label(data: LabelUpdate):
    
    # update SQLite
    with get_db_connection() as conn:
        cursor = conn.execute(""" UPDATE predictions SET actual_label = ? WHERE transaction_id = ? """,
                               (data.actual_label, data.transaction_id))
        conn.commit()

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Transaction ID not found in database")

    # update CSV
    try:
        update_label_in_csv(data.transaction_id, data.actual_label)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {"message": "Label updated successfully"}