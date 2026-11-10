import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "WELFake_Cleaned.csv"

df = pd.read_csv(DATA_FILE)

df["content"] = (
    df["title"].fillna("").astype(str)
    + " "
    + df["text"].fillna("").astype(str)
)

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Sample for initial investigation
train_sample = train_df.sample(
    n=min(5000, len(train_df)),
    random_state=42
)

test_sample = test_df.sample(
    n=min(1000, len(test_df)),
    random_state=42
)

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(train_sample["content"])
X_test = vectorizer.transform(test_sample["content"])

neighbors = NearestNeighbors(
    n_neighbors=1,
    metric="cosine",
    algorithm="brute"
)

neighbors.fit(X_train)

distances, indices = neighbors.kneighbors(X_test)

similarities = 1 - distances.flatten()

print("Train sample:", len(train_sample))
print("Test sample:", len(test_sample))

for threshold in [0.90, 0.95, 0.99]:

    count = (similarities >= threshold).sum()

    print(
        f"\nTest articles with training similarity >= {threshold}: "
        f"{count}"
    )

    print(
        "Percentage:",
        round(count / len(test_sample) * 100, 2),
        "%"
    )

print("\nTop 10 most similar cross-split pairs:")

for i in similarities.argsort()[-10:][::-1]:

    train_index = indices[i][0]

    print("\nSimilarity:", round(similarities[i], 4))

    print("Test title:", test_sample.iloc[i]["title"])
    print("Test label:", test_sample.iloc[i]["label"])

    print("Train title:", train_sample.iloc[train_index]["title"])
    print("Train label:", train_sample.iloc[train_index]["label"])