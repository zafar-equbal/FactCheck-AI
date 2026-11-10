import pandas as pd

df = pd.read_parquet(
    "data/train-00000-of-00001-290868f0a36350c5.parquet"
)

# Normalize article content
df["content"] = (
    df["title"].fillna("").astype(str).str.strip()
    + " "
    + df["text"].fillna("").astype(str).str.strip()
).str.strip()

# Group identical articles
groups = df.groupby("content")["label"].agg(
    count="size",
    unique_labels="nunique"
)

# Find duplicate articles
duplicates = groups[groups["count"] > 1]

# Find duplicate articles with conflicting labels
conflicts = duplicates[duplicates["unique_labels"] > 1]

print("Unique duplicate articles:", len(duplicates))
print("Duplicate records beyond first occurrence:",
      int((duplicates["count"] - 1).sum()))

print("Conflicting duplicate articles:", len(conflicts))

print("\nDuplicate article distribution by label:")

print(
    df[df["content"].isin(duplicates.index)]
    .groupby("label")
    .size()
)