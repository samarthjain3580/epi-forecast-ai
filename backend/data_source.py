import requests
import pandas as pd

def fetch_data(country="india"):
    url = f"https://disease.sh/v3/covid-19/historical/{country}?lastdays=all"
    res = requests.get(url).json()

    cases = res["timeline"]["cases"]

    df = pd.DataFrame(list(cases.items()), columns=["date", "cases"])

    # ✅ Convert to daily cases
    df["cases"] = df["cases"].diff().fillna(0)

    # ✅ FIX 1: remove negative values
    df["cases"] = df["cases"].clip(lower=0)

    # ✅ FIX 2: smooth extreme spikes (optional but good)
    df["cases"] = df["cases"].rolling(window=3, min_periods=1).mean()

    return df