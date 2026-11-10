import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "WELFake_Dataset.csv"

df = pd.read_csv(DATA_FILE)

patterns = [
    "For entire story:",
    "Via:",
    "Source:",
    "Originally published",
    "According to Reuters",
    "reported by Reuters",
    "The New York Times reported",
    "CNN reported"
]

for pattern in patterns:

    matches = df[
        df["text"].fillna("").str.contains(
            pattern,
            case=False,
            regex=False
        )
    ]

    print("\n" + "=" * 50)
    print("PATTERN:", pattern)
    print("Total matches:", len(matches))

    print("\nLabel distribution:")
    print(matches["label"].value_counts().sort_index())

    for _, row in matches.head(3).iterrows():
        print("\nTitle:", row["title"])
        print("Label:", row["label"])