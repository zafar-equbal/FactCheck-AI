import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "WELFake_Dataset.csv"

df = pd.read_csv(DATA_FILE)

df["content"] = (
    df["title"].fillna("").astype(str)
    + " "
    + df["text"].fillna("").astype(str)
).str.lower().str.strip()

# Group identical content
groups = df.groupby("content")["label"].nunique()

# Find identical content assigned different labels
conflicts = groups[groups > 1]

print("Total articles:", len(df))
print("Unique articles:", df["content"].nunique())
print("Conflicting duplicate groups:", len(conflicts))

if len(conflicts) > 0:
    print("\nExamples of conflicting articles:")

    for content in conflicts.index[:5]:
        print("\nArticle:", content[:300])
        print(
            df[df["content"] == content][["title", "label"]]
        )
        