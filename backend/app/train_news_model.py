
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE = Path(__file__).resolve().parents[1]

DATA = BASE / "data" / "cleaned_news.csv"
MODEL_DIR = BASE / "models"
MODEL_PATH = MODEL_DIR / "news_pipeline.joblib"


def main():
    # 1. Load the cleaned dataset
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA)

    required_columns = {"title", "text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns: {required_columns}"
        )

    # 2. Clean the data
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    # Combine headline and article text
    df["content"] = (
        df["title"] + " " + df["text"]
    ).str.strip()

    # Remove rows without usable text
    df = df[df["content"].str.len() >= 20].copy()

    # Remove exact duplicate content
    df = df.drop_duplicates(subset=["content"])

    if len(df) < 10 or df["label"].nunique() != 2:
        raise ValueError(
            "Need at least 10 usable rows and both labels (0 and 1)."
        )

    print("Dataset rows:", len(df))
    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    # 3. Split before training to evaluate on unseen examples
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    # 4. Build the model
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=30000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    # 5. Train only on the training split
    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    # 6. Evaluate on the held-out test split
    predictions = pipeline.predict(X_test)

    print("\n===== MODEL EVALUATION =====")
    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Fake (0)", "Real (1)"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # 7. Save the trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"\nSaved model: {MODEL_PATH}")
    print("Label mapping: 0 = Fake, 1 = Real")
    print(
        "Note: These metrics measure performance on this dataset's "
        "held-out split, not guaranteed real-world fact-checking accuracy."
    )


if __name__ == "__main__":
    main()

