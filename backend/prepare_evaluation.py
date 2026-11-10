import pandas as pd
from sklearn.model_selection import train_test_split

# Load original dataset
df = pd.read_parquet(
    "data/train-00000-of-00001-290868f0a36350c5.parquet"
)

# Remove missing values
df = df.dropna(subset=["title", "text", "label"]).copy()

# Combine title and article text
df["content"] = (
    df["title"].astype(str) + " " + df["text"].astype(str)
)

# Remove empty articles
df = df[df["content"].str.strip().str.len() >= 20]

# Select 1,000 stratified samples
evaluation_df, _ = train_test_split(
    df,
    train_size=1000,
    stratify=df["label"],
    random_state=42
)

# Save evaluation dataset
evaluation_df.to_csv(
    "data/evaluation_dataset.csv",
    index=False
)

print("Evaluation dataset created!")
print("Total articles:", len(evaluation_df))
print("\nLabel distribution:")
print(evaluation_df["label"].value_counts())

print("\nSaved: data/evaluation_dataset.csv")