from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "news.csv"
MODEL_DIR = BASE / "models"
MODEL_PATH = MODEL_DIR / "news_pipeline.joblib"

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV needs columns named 'text' and 'label'.")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[(df.text.str.len() >= 20) & df.label.isin(["fake", "real"])]
    if len(df) < 10 or df.label.nunique() < 2:
        raise ValueError("Need at least 10 rows and both labels: fake, real.")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=30000, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipeline.fit(df.text, df.label)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Training rows: {len(df)}")
    print("Demo starter only: evaluate with a separate held-out dataset before reporting performance.")

if __name__ == "__main__":
    main()
