import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "WELFake_Cleaned.csv"

df = pd.read_csv(DATA_FILE)

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

X_title = df["title"]
X_text = df["text"]
y = df["label"]

X_title_train, X_title_test, y_train, y_test = train_test_split(
    X_title, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_text_train, X_text_test, _, _ = train_test_split(
    X_text, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

def evaluate_model(X_train, X_test, y_train, y_test, name):

    vectorizer = TfidfVectorizer(
        max_features=50000,
        stop_words="english",
        ngram_range=(1, 2)
    )

    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    model.fit(train_vectors, y_train)

    predictions = model.predict(test_vectors)

    print(f"\n{'=' * 50}")
    print(name)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

evaluate_model(
    X_title_train,
    X_title_test,
    y_train,
    y_test,
    "TITLE ONLY MODEL"
)

evaluate_model(
    X_text_train,
    X_text_test,
    y_train,
    y_test,
    "ARTICLE TEXT ONLY MODEL"
)