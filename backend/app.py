import os
from flask import Flask, jsonify, request
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 🔥 IMPORTANT FIX (relative import)
from backend.data_source import fetch_data

app = Flask(__name__)

# ✅ MODEL PATH (robust for both local + render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "lstm_model_fixed.h5")

# ✅ LOAD MODEL SAFELY
try:
    model = load_model(model_path, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", str(e))
    model = None


# ✅ ROOT ROUTE (health check)
@app.route("/")
def home():
    return "API running 🚀"


# ✅ PREDICT ROUTE
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

        last_14 = values[-14:]
        last_14_scaled = scaler.transform(last_14)

        current_input = last_14_scaled.reshape(1, 14, 1)

        predictions = []

        for _ in range(7):
            pred = model.predict(current_input, verbose=0)
            predictions.append(pred[0][0])

            pred_reshaped = pred.reshape(1, 1, 1)

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


# ✅ RUN (Render compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)