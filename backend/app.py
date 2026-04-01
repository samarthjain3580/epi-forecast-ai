import os
from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from backend.data_source import fetch_data

app = Flask(__name__)

# ✅ model initially None
model = None

# ✅ correct path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "lstm_model_fixed.h5")


def load_my_model():
    global model
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False
        )
        print("Model loaded ✅")


@app.route("/")
def home():
    return "API running 🚀"


@app.route("/predict")
def predict():
    try:
        # 🔥 LOAD MODEL ONLY WHEN NEEDED
        load_my_model()

        country = request.args.get("country", "india")
        disease = request.args.get("disease", "covid")

        df = fetch_data(country)

        if disease == "flu":
            df["cases"] *= 0.6
        elif disease == "dengue":
            df["cases"] *= 0.3

        values = df["cases"].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(values[-60:])

        past_30 = values[-30:].flatten().tolist()

        last_14 = values[-14:]
        last_14_scaled = scaler.transform(last_14)

        if last_14_scaled.max() == 0:
            last_14_scaled += 1e-6

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