import pandas as pd
import matplotlib.pyplot as plt

# load actual data
actual = pd.read_csv("data/processed/india_daily_cases.csv", index_col=0)

# load predictions (example placeholders for now)
arima_pred = actual.tail(100) * 0.9
lstm_pred = actual.tail(100) * 1.05

plt.figure(figsize=(10,5))

plt.plot(actual.tail(100), label="Actual")
plt.plot(arima_pred, label="ARIMA Prediction")
plt.plot(lstm_pred, label="LSTM Prediction")

plt.title("ARIMA vs LSTM Comparison")
plt.legend()
plt.show()