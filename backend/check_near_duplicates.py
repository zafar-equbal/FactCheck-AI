import pandas as pd
from pathlib import Path
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

# Use a sample to keep computation manageable
sample = df.sample(
    n=min(5000, len(df)),
    random_state=42
).reset_index(drop=True)

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(sample["content"])

neighbors = NearestNeighbors(
    n_neighbors=2,
    metric="cosine",
    algorithm="brute"
)

neighbors.fit(X)

distances, indices = neighbors.kneighbors(X)

threshold = 0.90
count = 0

print("Checking near-duplicate articles...")
print("Similarity threshold:", threshold)

for i in range(len(sample)):

    similarity = 1 - distances[i][1]

    if similarity >= threshold:

        j = indices[i][1]

        print("\nSimilarity:", round(similarity, 4))
        print("Article 1:", sample.iloc[i]["title"])
        print("Label 1:", sample.iloc[i]["label"])

        print("Article 2:", sample.iloc[j]["title"])
        print("Label 2:", sample.iloc[j]["label"])

        count += 1

        if count >= 15:
            break

print("\nNear-duplicate pairs displayed:", count)