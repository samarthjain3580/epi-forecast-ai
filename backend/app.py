from flask import Flask, jsonify, request
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from data_source import fetch_data

app = Flask(__name__)

model = load_model("models/lstm_model.h5", compile=False)

@app.route("/predict")
def predict():

    country = request.args.get("country", "india")
    disease = request.args.get("disease", "covid")

    df = fetch_data(country)

    # simulate diseases
    if disease == "flu":
        df["cases"] *= 0.6
    elif disease == "dengue":
        df["cases"] *= 0.3

    values = df["cases"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaler.fit(values)

    # past 30 days
    past_30 = values[-30:].flatten().tolist()

    # last 14 days input
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

if __name__ == "__main__":
    app.run(debug=True)