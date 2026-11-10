import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Model directory
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models" / "factcheck_distilbert"


# Device configuration
DEVICE = torch.device("cpu")


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True
)


# Load trained DistilBERT model
model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True
)

model.to(DEVICE)
model.eval()


def predict_news(text: str):

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # Generate prediction
    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    predicted_class = torch.argmax(probabilities).item()

    labels = {
        0: "FAKE",
        1: "REAL"
    }

    return {
        "prediction": labels[predicted_class],
        "confidence": round(
            probabilities[predicted_class].item(),
            4
        ),
        "fake_probability": round(
            probabilities[0].item(),
            4
        ),
        "real_probability": round(
            probabilities[1].item(),
            4
        )
    }