import pandas as pd

PATH = "data/raw/time_series_covid19_confirmed_global.csv"

df = pd.read_csv(PATH)

# keep only date columns (everything after Lat, Long)
date_cols = df.columns[4:]

# group by country and sum cases
df_country = df.groupby("Country/Region")[date_cols].sum()

# transpose so dates become rows
ts = df_country.T
ts.index = pd.to_datetime(ts.index)

# example: India daily cases
india = ts["India"].diff().fillna(0)

india.to_csv("data/processed/india_daily_cases.csv")

print("✅ Preprocessed daily cases saved")
