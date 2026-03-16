import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
data = pd.read_csv("data/processed/india_daily_cases.csv", index_col=0)
values = data.values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)

# Create sequences
X = []
y = []
window = 14

for i in range(window, len(scaled)):
    X.append(scaled[i-window:i, 0])
    y.append(scaled[i, 0])

X = np.array(X)
y = np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1],1)))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

# Train model
model.fit(X, y, epochs=10, batch_size=16)

# Predict
predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)

actual = scaler.inverse_transform(y.reshape(-1,1))

# Plot
plt.figure(figsize=(10,5))
plt.plot(actual, label="Actual")
plt.plot(predicted, label="Predicted")
plt.legend()
plt.title("LSTM Prediction")
plt.show()