import os
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Ensure project root is in sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.data_source import fetch_data
from backend.model_numpy import LSTMPredictor

app = Flask(__name__)
CORS(app)

# LOAD MODEL SAFELY
try:
    model = LSTMPredictor()
    print("Model loaded successfully (NumPy)")
except Exception as e:
    print("Model loading failed:", str(e))
    model = None


# ROOT ROUTE (health check)
@app.route("/")
def home():
    return "API running"


# PREDICT ROUTE
@app.route("/predict")
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        country = request.args.get("country", "india")
        disease = request.args.get("disease", "covid")

        df = fetch_data(country)

        # disease scaling
        if disease == "flu":
            df["cases"] *= 0.6
        elif disease == "dengue":
            df["cases"] *= 0.3

        values = df["cases"].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        scaler.fit(values)

        past_30 = values[-30:].flatten().tolist()

        last_30 = values[-30:]
        last_30_scaled = scaler.transform(last_30)

        current_input = last_30_scaled.reshape(1, 30, 1).astype(np.float32)

        predictions = []

        for _ in range(7):
            pred = model.predict(current_input)
            pred_val = float(pred[0][0])
            predictions.append(pred_val)

            pred_reshaped = np.array([[[pred_val]]], dtype=np.float32)

            current_input = np.append(
                current_input[:, 1:, :],
                pred_reshaped,
                axis=1
            )

        predictions = np.array(predictions).reshape(-1, 1)
        real_predictions = scaler.inverse_transform(predictions)

        result = [round(max(0, x), 2) for x in real_predictions.flatten().tolist()]

        return jsonify({
            "country": country,
            "disease": disease,
            "past_30_days": past_30,
            "predicted_cases_next_7_days": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# RUN (Render compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
