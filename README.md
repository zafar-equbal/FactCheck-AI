# FactCheck AI

Starter project: React + Vite frontend, Python + FastAPI backend, and a TF-IDF + Logistic Regression news classifier.

> The included CSV is tiny demo data only. It is not reliable for real-world predictions. Replace it with a reputable dataset, evaluate on held-out data, and describe limitations. A model prediction is not proof that a claim is true or false.

## Requirements
- Python 3.10+
- Node.js 20+ recommended
- VS Code

## Run backend (VS Code terminal 1, Windows PowerShell)
```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.train_news_model
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.
API docs: http://127.0.0.1:8000/docs

## Run frontend (terminal 2)
```powershell
cd frontend
npm install
npm run dev
```
Open the URL Vite prints, usually http://localhost:5173.

## Dataset format
`backend/data/news.csv` must contain `text` and `label` columns; labels must be `fake` or `real`. Replace the illustrative rows with a reputable, properly licensed dataset before evaluating or presenting results. Avoid duplicate articles across train/test splits. Report precision, recall, F1-score and confusion matrix, not accuracy alone.

## Included / not yet implemented
- Included: React dashboard, FastAPI health endpoint, text classification endpoint, model training script.
- Placeholder only: image/video UI and deepfake API endpoint. Deepfake model, media processing, database/history, URL extraction and explainability are future milestones.
