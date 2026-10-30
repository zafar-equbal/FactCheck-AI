
from pathlib import Path

import pandas as pd

# Locate the backend directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "train-00000-of-00001-290868f0a36350c5.parquet"
OUTPUT_FILE = DATA_DIR / "cleaned_news.csv"

# Load the original dataset
df = pd.read_parquet(INPUT_FILE)

print("Original dataset shape:", df.shape)

# Keep only the columns needed for news detection
df = df[["title", "text", "label"]].copy()

# Replace missing title and text values
df["title"] = df["title"].fillna("").astype(str).str.strip()
df["text"] = df["text"].fillna("").astype(str).str.strip()

# Remove records where both title and text are empty
df = df[(df["title"] != "") | (df["text"] != "")].copy()

# Ensure labels are numeric and valid
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df[df["label"].isin([0, 1])].copy()
df["label"] = df["label"].astype(int)

# Remove duplicate records
df = df.drop_duplicates(subset=["title", "text"])

# Save the cleaned dataset
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("\nDataset preparation complete!")
print("Cleaned dataset shape:", df.shape)
print("Label distribution:")
print(df["label"].value_counts().sort_index())
print("\nSaved to:", OUTPUT_FILE)

