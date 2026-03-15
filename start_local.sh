#!/bin/bash
# Start FastAPI backend
echo "Starting FastAPI backend..."
uvicorn task_2_credit_card_fraud_detecation.api.api:app --reload --port 8000 &
BACKEND_PID=$!

# Start Streamlit frontend
echo "Starting Streamlit frontend..."
streamlit run task_2_credit_card_fraud_detecation/app/app.py --server.port 8501 &
FRONTEND_PID=$!

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID


# ./start_local.sh
