# Epidemic Forecast AI – Workflow

## System Architecture

Streamlit Dashboard (Frontend)
        ↓
Flask API (Backend)
        ↓
LSTM Model Prediction
        ↓
Return Results to Dashboard

---

## Project Pipeline

1. Data Collection
   - COVID time‑series dataset stored in `data/raw`

2. Data Preprocessing
   - Script: `backend/preprocess.py`
   - Converts cumulative data → daily cases

3. Forecasting Models
   - ARIMA Model → `backend/arima_model.py`
   - LSTM Model → `backend/lstm_model.py`

4. Model Storage
   - Saved model in `models/lstm_model.h5`

5. Prediction API
   - Flask API → `backend/app.py`
   - Endpoint:
     ```
     http://127.0.0.1:5000/predict
     ```

6. Dashboard
   - Streamlit app → `frontend/dashboard.py`
   - Runs on:
     ```
     http://localhost:8501
     ```

---

## How to Run the Project

### Step 1 – Activate environment

### Step 2 – Start Flask API

### Step 3 – Start Streamlit Dashboard

Open a new terminal:

### Step 4 – Open Dashboard
http://localhost:8501

Click **Get Prediction** to see epidemic forecasts.

---

## Technologies Used

- Python
- Pandas
- TensorFlow / Keras
- ARIMA (statsmodels)
- Flask API
- Streamlit Dashboard