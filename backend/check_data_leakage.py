import pandas as pd
from sklearn.model_selection import train_test_split

# Load original dataset
path = "data/train-00000-of-00001-290868f0a36350c5.parquet"
df = pd.read_parquet(path)

# Reproduce original 80:10:10 split
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

# Normalize article content
def get_content(data):
    return (
        data["title"].fillna("").astype(str).str.strip()
        + " "
        + data["text"].fillna("").astype(str).str.strip()
    ).str.strip()

train_content = set(get_content(train_df))
test_content = set(get_content(test_df))

overlap = train_content.intersection(test_content)

print("Training records:", len(train_df))
print("Testing records:", len(test_df))
print("Unique training articles:", len(train_content))
print("Unique testing articles:", len(test_content))
print("Articles appearing in both:", len(overlap))