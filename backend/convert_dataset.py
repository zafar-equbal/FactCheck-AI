import pandas as pd
from pathlib import Path

INPUT_FILE = Path(
    "data/train-00000-of-00001-290868f0a36350c5.parquet"
)

OUTPUT_FILE = Path("data/WELFake_Dataset.csv")

print("Loading Parquet dataset...")

df = pd.read_parquet(INPUT_FILE)

print("Original shape:", df.shape)

# Save as CSV without changing labels or records
df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\nConversion completed!")
print("CSV shape:", df.shape)
print("Saved at:", OUTPUT_FILE)
print("Label distribution:")
print(df["label"].value_counts().sort_index())