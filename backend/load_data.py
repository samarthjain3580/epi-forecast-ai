import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    print("✅ Dataset loaded")
    print(df.head())
    print("\nShape:", df.shape)
    return df

if __name__ == "__main__":
    path = "../data/raw/time_series_covid19_confirmed_global.csv"
    load_dataset(path)