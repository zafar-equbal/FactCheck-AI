import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "WELFake_Dataset.csv"

df = pd.read_csv(DATA_FILE)

for label in [0, 1]:
    print(f"\n{'=' * 60}")
    print(f"LABEL: {label}")
    print(f"{'=' * 60}")

    samples = df[df["label"] == label].sample(
        n=20,
        random_state=42
    )

    for index, row in samples.iterrows():
        print(f"\nTitle: {row['title']}")
        print(f"Label: {row['label']}")