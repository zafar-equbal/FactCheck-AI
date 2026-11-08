from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent

DATA_PATH = BASE / "data" / "cleaned_news.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

print("Total records:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nLabel percentages:")
print(df["label"].value_counts(normalize=True) * 100)


# Combine title and text
df["content"] = (
    df["title"].fillna("").astype(str)
    + " "
    + df["text"].fillna("").astype(str)
)


print("\n========== DUPLICATE ANALYSIS ==========")

print("Duplicate titles:", df["title"].duplicated().sum())

print("Duplicate text:", df["text"].duplicated().sum())

print("Duplicate combined content:",
      df["content"].duplicated().sum())


print("\n========== SOURCE WORD ANALYSIS ==========")

source_words = [
    "reuters",
    "breitbart",
    "cnn",
    "bbc",
    "new york times",
    "washington post",
    "associated press",
    "getty images",
    "twitter"
]

for word in source_words:

    matches = df[
        df["content"].str.contains(
            word,
            case=False,
            regex=False
        )
    ]

    fake_count = (matches["label"] == 0).sum()
    real_count = (matches["label"] == 1).sum()

    print(f"\n{word.upper()}")
    print("Total:", len(matches))
    print("Fake:", fake_count)
    print("Real:", real_count)


print("\n========== SAMPLE ARTICLES ==========")

for label, name in [(0, "FAKE"), (1, "REAL")]:

    sample = df[df["label"] == label].head(3)

    print(f"\n{name} NEWS SAMPLES:")

    for _, row in sample.iterrows():

        print("\nTitle:", row["title"])

        print("Text:", str(row["text"])[:300])



print("\n========== REUTERS ARTICLE SAMPLES ==========")

reuters_articles = df[
    df["content"].str.contains(
        "reuters",
        case=False,
        regex=False
    )
]

for label, category in [(0, "FAKE"), (1, "REAL")]:

    print(f"\n{category} REUTERS ARTICLES:")

    samples = reuters_articles[
        reuters_articles["label"] == label
    ].sample(
        min(5, len(reuters_articles[reuters_articles["label"] == label])),
        random_state=42
    )

    for _, row in samples.iterrows():

        print("\nTitle:", row["title"])

        print("Article:", str(row["text"])[:500])



print("\n========== REUTERS ARTICLE SAMPLES ==========")

reuters_articles = df[
    df["content"].str.contains(
        "reuters",
        case=False,
        regex=False
    )
]

for label, category in [(0, "FAKE"), (1, "REAL")]:

    print(f"\n{category} REUTERS ARTICLES:")

    samples = reuters_articles[
        reuters_articles["label"] == label
    ].sample(
        min(5, len(reuters_articles[reuters_articles["label"] == label])),
        random_state=42
    )

    for _, row in samples.iterrows():

        print("\nTitle:", row["title"])

        print("Article:", str(row["text"])[:500])        
