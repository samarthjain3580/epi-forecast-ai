import requests
import pandas as pd

def fetch_data(country="india"):
    url = f"https://disease.sh/v3/covid-19/historical/{country}?lastdays=all"
    res = requests.get(url).json()

    cases = res["timeline"]["cases"]

    df = pd.DataFrame(list(cases.items()), columns=["date", "cases"])
    df["cases"] = df["cases"].diff().fillna(0)

    return df