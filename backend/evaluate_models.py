import pandas as pd
import joblib
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------
# 1. Load evaluation dataset
# ------------------------------------------------

df = pd.read_csv("data/evaluation_dataset.csv")

texts = df["content"].astype(str).tolist()
y_true = df["label"].astype(int).tolist()

print("Evaluation articles:", len(df))

# ------------------------------------------------
# 2. Evaluate Logistic Regression
# ------------------------------------------------

print("\nEvaluating Logistic Regression...")

pipeline = joblib.load("models/news_pipeline.joblib")

lr_predictions = pipeline.predict(texts).astype(int)

# ------------------------------------------------
# 3. Load DistilBERT
# ------------------------------------------------

print("\nLoading DistilBERT...")

MODEL_DIR = Path("models/factcheck_distilbert")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True
)

device = torch.device("cpu")
model.to(device)
model.eval()

# ------------------------------------------------
# 4. Batch prediction
# ------------------------------------------------

print("\nEvaluating DistilBERT...")

distilbert_predictions = []

for start in range(0, len(texts), 16):

    batch = texts[start:start + 16]

    inputs = tokenizer(
        batch,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    predictions = torch.argmax(
        outputs.logits,
        dim=1
    ).cpu().tolist()

    distilbert_predictions.extend(predictions)

    print(
        f"Processed {min(start + 16, len(texts))}/{len(texts)}"
    )

# ------------------------------------------------
# 5. Evaluation function
# ------------------------------------------------

def evaluate(name, predictions):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Accuracy:", round(
        accuracy_score(y_true, predictions), 4
    ))

    print("Precision:", round(
        precision_score(y_true, predictions, zero_division=0), 4
    ))

    print("Recall:", round(
        recall_score(y_true, predictions, zero_division=0), 4
    ))

    print("F1 Score:", round(
        f1_score(y_true, predictions, zero_division=0), 4
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["FAKE", "REAL"],
        zero_division=0
    ))


# ------------------------------------------------
# 6. Compare both models
# ------------------------------------------------

evaluate("LOGISTIC REGRESSION", lr_predictions)

evaluate("DISTILBERT", distilbert_predictions)

# ------------------------------------------------
# 7. Save predictions
# ------------------------------------------------

df["logistic_prediction"] = lr_predictions
df["distilbert_prediction"] = distilbert_predictions

df.to_csv(
    "data/evaluation_results.csv",
    index=False
)

print("\nEvaluation completed!")
print("Results saved to data/evaluation_results.csv")