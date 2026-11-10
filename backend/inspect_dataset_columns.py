import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "WELFake_Dataset.csv"

df = pd.read_csv(DATA_FILE)

print("Dataset columns:")
print(df.columns.tolist())

print("\nDataset information:")
print(df.info())

print("\nSample records:")
print(df.head(5).to_string())