from flask import Flask, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("models/lstm_model.h5", compile=False)

@app.route("/predict")
def predict():

    # example input sequence
    sample_input = np.random.rand(1,14,1)

    prediction = model.predict(sample_input)

    result = prediction.flatten().tolist()

    return jsonify({
        "predicted_cases": result
    })

if __name__ == "__main__":
    app.run(debug=True)