import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "WELFake_Cleaned.csv"

df = pd.read_csv(DATA_FILE)

df["text"] = df["text"].fillna("").str.lower()

y = df["label"]

patterns = [
    "for entire story:",
    "via:",
    "source:",
    "originally published",
    "reported by reuters"
]

X = df["text"].apply(
    lambda text: " ".join(
        pattern for pattern in patterns if pattern in text
    )
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

vectorizer = CountVectorizer(binary=True)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

print("SOURCE SHORTCUT MODEL")
print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nFeatures learned:")
print(vectorizer.get_feature_names_out())