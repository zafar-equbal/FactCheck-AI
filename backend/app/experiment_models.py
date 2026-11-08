from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


BASE = Path(__file__).resolve().parents[1]

DATA_PATH = BASE / "data" / "cleaned_news.csv"

df = pd.read_csv(DATA_PATH)

df["title"] = df["title"].fillna("").astype(str)
df["text"] = df["text"].fillna("").astype(str)

df["content"] = df["title"] + " " + df["text"]

df = df[df["content"].str.len() >= 20]
df = df.drop_duplicates(subset=["content"])

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42
    )
}

print("\n========== SOURCE BIAS EXPERIMENT ==========")

test_article = """
Scientists at the International Space Research Institute
have announced that the Moon will disappear from Earth's sky
for exactly seven days next month. A secret gravitational
experiment will temporarily make the Moon invisible.
"""

variations = {
    "Original": test_article,

    "Without publisher names": test_article.replace(
        "Reuters", ""
    ).replace("Getty", ""),

    "Without formatting artifacts": " ".join(
        test_article.replace("\n", " ").split()
    )
}
print("\n========== SOURCE BIAS EXPERIMENT ==========")

test_article = """
Scientists at the International Space Research Institute
have announced that the Moon will disappear from Earth's sky
for exactly seven days next month. A secret gravitational
experiment will temporarily make the Moon invisible.
"""

variations = {
    "Original": test_article,

    "Without publisher names": test_article.replace(
        "Reuters", ""
    ).replace("Getty", ""),

    "Without formatting artifacts": " ".join(
        test_article.replace("\n", " ").split()
    )
}


for name, classifier in models.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=30000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True
            )
        ),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    print("\nSOURCE BIAS TEST:")

for variation_name, article in variations.items():

    prediction = pipeline.predict([article])[0]

    print(
        variation_name,
        "->",
        "FAKE" if prediction == 0 else "REAL"
    )

    predictions = pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    test_articles = [
        """
        Scientists at the International Space Research Institute
        have announced that the Moon will disappear from Earth's sky
        for exactly seven days next month. A secret gravitational
        experiment will temporarily make the Moon invisible.
        """,

        """
        NASA successfully launched the James Webb Space Telescope
        on December 25, 2021. The telescope was developed through
        international collaboration and observes distant galaxies
        using infrared technology.
        """
    ]

    print("\nCustom Article Predictions:")

    for article in test_articles:
        prediction = pipeline.predict([article])[0]

        print("Prediction:", "FAKE" if prediction == 0 else "REAL")


print("\n========== 100 ARTICLE COMPARISON ==========")

for label, category in [(0, "FAKE"), (1, "REAL")]:

    samples = df[df["label"] == label].sample(
        n=100,
        random_state=42
    )

    print(f"\n{category} NEWS:")

    for model_name, classifier in models.items():

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=30000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True
                )
            ),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(
            samples["content"]
        )

        correct = (predictions == label).sum()

        print(
            f"{model_name}: "
            f"{correct}/100 correct"
        )        