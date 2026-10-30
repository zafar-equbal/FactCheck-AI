from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "news_pipeline.joblib"

app = FastAPI(title="FactCheck AI API", version="0.1.0",
              description="Starter API for AI-assisted news and media analysis.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

class NewsRequest(BaseModel):
    text: str = Field(min_length=20, max_length=20000)

@app.get("/api/health")
def health():
    return {"status": "ok", "project": "FactCheck AI",
            "news_model_ready": MODEL_PATH.exists(), "deepfake_model_ready": False}

@app.post("/api/analyze/news")
def analyze_news(request: NewsRequest):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503,
            detail="News model is not trained. Run: python -m app.train_news_model")
    try:
        model = joblib.load(MODEL_PATH)
        probabilities = model.predict_proba([request.text])[0]
        classes = list(model.classes_)
        idx = int(probabilities.argmax())
        return {
            "status": "ok",
            "prediction": str(classes[idx]),
            "confidence": round(float(probabilities[idx]), 4),
            "disclaimer": "Model prediction only—not proof of truth or authenticity. Verify important claims using reliable independent sources."
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model error: {exc}") from exc

@app.post("/api/analyze/media")
def analyze_media_placeholder():
    raise HTTPException(status_code=501,
        detail="Deepfake detection is planned but not implemented in this starter.")
