from pathlib import Path
import joblib

BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "models" / "news_pipeline.joblib"

model = joblib.load(MODEL_PATH)

test_articles = [
    {
        "name": "Known Real Event",
        "text": """
        NASA successfully launched the James Webb Space Telescope
        on December 25, 2021. The telescope was developed through
        international collaboration involving NASA, ESA and CSA.
        """
    },
    {
        "name": "Fabricated Moon Story",
        "text": """
        Scientists have announced that the Moon will disappear
        from Earth's sky for exactly seven days next month.
        A secret gravitational experiment will temporarily make
        the Moon invisible to everyone on the planet.
        """
    }
]

for article in test_articles:

    probabilities = model.predict_proba([article["text"]])[0]

    classes = list(model.classes_)

    index = probabilities.argmax()

    prediction = classes[index]

    confidence = probabilities[index] * 100

    label = "FAKE NEWS" if prediction == 0 else "REAL NEWS"

    print("\n------------------------------")
    print("Article:", article["name"])
    print("Prediction:", label)
    print(f"Confidence: {confidence:.2f}%")

    print("\nClass probabilities:")

    for cls, probability in zip(classes, probabilities):
        print(f"Class {cls}: {probability * 100:.2f}%")



print("\n========== MODEL ANALYSIS ==========")

vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

feature_names = vectorizer.get_feature_names_out()
coefficients = classifier.coef_[0]

# Highest positive coefficients indicate REAL class
top_real = coefficients.argsort()[-20:][::-1]

# Lowest negative coefficients indicate FAKE class
top_fake = coefficients.argsort()[:20]

print("\nWords most associated with REAL NEWS:")

for index in top_real:
    print(f"{feature_names[index]}: {coefficients[index]:.4f}")

print("\nWords most associated with FAKE NEWS:")

for index in top_fake:
    print(f"{feature_names[index]}: {coefficients[index]:.4f}")


import pandas as pd
import joblib
from pathlib import Path

print("\n========== EXACT DATASET TEST ==========")

BASE = Path(__file__).resolve().parent

df = pd.read_csv(BASE / "data" / "cleaned_news.csv")
model = joblib.load(BASE / "models" / "news_pipeline.joblib")

# Select one exact Fake News record
sample = df[df["label"] == 0].iloc[0]

title = str(sample["title"])
article = str(sample["text"])

# Prediction using article body only
body_prediction = model.predict([article])[0]
body_probabilities = model.predict_proba([article])[0]

# Prediction using title + article
full_content = title + " " + article

full_prediction = model.predict([full_content])[0]
full_probabilities = model.predict_proba([full_content])[0]

print("\nExpected Label: FAKE (0)")
print("\nTitle:", title)

print("\n--- BODY ONLY ---")
print("Prediction:", body_prediction)
print("Probabilities:", body_probabilities)

print("\n--- TITLE + ARTICLE ---")
print("Prediction:", full_prediction)
print("Probabilities:", full_probabilities)   



import requests

print("\n========== FASTAPI TEST ==========")

response = requests.post(
    "http://127.0.0.1:8000/api/analyze/news",
    json={"text": full_content}
)

print("Status Code:", response.status_code)
print("API Response:", response.json())


print("\n========== BATCH DATASET TEST ==========")

import pandas as pd
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent

df = pd.read_csv(BASE / "data" / "cleaned_news.csv")
model = joblib.load(BASE / "models" / "news_pipeline.joblib")

for label, category in [(0, "FAKE"), (1, "REAL")]:

    samples = df[df["label"] == label].sample(
        n=10,
        random_state=42
    )

    correct = 0

    print(f"\n{category} NEWS TEST:")

    for _, row in samples.iterrows():

        content = str(row["title"]) + " " + str(row["text"])

        prediction = int(model.predict([content])[0])

        if prediction == label:
            correct += 1

        result = "PASS" if prediction == label else "FAIL"

        print(
            f"{result} | Expected: {label} | "
            f"Predicted: {prediction} | {row['title'][:70]}"
        )

    print(f"\nAccuracy: {correct}/10")




    print("\n========== UNSEEN TEST DATA ==========")

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BASE = Path(__file__).resolve().parent

df = pd.read_csv(BASE / "data" / "cleaned_news.csv")

df["content"] = (
    df["title"].fillna("").astype(str)
    + " "
    + df["text"].fillna("").astype(str)
).str.strip()

df = df[df["content"].str.len() >= 20]
df = df.drop_duplicates(subset=["content"])

X = df["content"]
y = df["label"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = joblib.load(BASE / "models" / "news_pipeline.joblib")

predictions = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))