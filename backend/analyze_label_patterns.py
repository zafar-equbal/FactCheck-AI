import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "WELFake_Dataset.csv"

df = pd.read_csv(DATA_FILE)

for label in [0, 1]:

    subset = df[df["label"] == label]

    print("\n" + "=" * 60)
    print(f"LABEL {label}")
    print("=" * 60)

    print("Total articles:", len(subset))

    print("\nTitle examples:")
    for title in subset["title"].dropna().head(15):
        print("-", title)

    print("\nArticles containing common publisher names:")

    publishers = [
        "Reuters",
        "New York Times",
        "Breitbart",
        "CNN",
        "BBC",
        "Fox News",
        "Washington Post",
        "Guardian"
    ]

    text = subset["title"].fillna("") + " " + subset["text"].fillna("")

    for publisher in publishers:
        count = text.str.contains(
            publisher,
            case=False,
            regex=False
        ).sum()

        print(f"{publisher}: {count}")