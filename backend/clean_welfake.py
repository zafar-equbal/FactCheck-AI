import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "WELFake_Dataset.csv"
OUTPUT_FILE = DATA_DIR / "WELFake_Cleaned.csv"

print("Loading WELFake dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# Keep required columns
df = df[["title", "text", "label"]].copy()

# Clean missing values
df["title"] = df["title"].fillna("").astype(str).str.strip()
df["text"] = df["text"].fillna("").astype(str).str.strip()

# Remove articles without meaningful content
df = df[
    (df["title"] != "") | (df["text"] != "")
].copy()

# Validate labels
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df[df["label"].isin([0, 1])].copy()
df["label"] = df["label"].astype(int)

# Combine title and text for duplicate detection
df["content"] = (
    df["title"] + " " + df["text"]
).str.strip()

# Remove exact duplicate articles
before = len(df)

df = df.drop_duplicates(
    subset=["content"],
    keep="first"
).copy()

removed = before - len(df)

# Save cleaned dataset
df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\nCleaning completed!")
print("Original records:", before)
print("Duplicates removed:", removed)
print("Final records:", len(df))

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())

print("\nSaved to:", OUTPUT_FILE)