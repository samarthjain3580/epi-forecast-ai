import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv("data/processed/india_daily_cases.csv", index_col=0, parse_dates=True)

model = ARIMA(data, order=(5,1,0))
fit = model.fit()

forecast = fit.forecast(steps=14)
print("Next 14 days forecast:\n", forecast)

data.plot(label="Actual")
forecast.plot(label="Forecast", style="--")
plt.legend()
plt.show()
