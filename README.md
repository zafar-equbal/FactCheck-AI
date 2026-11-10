# 🔍 FactCheck AI
### AI-Powered Fake News Detection System

FactCheck AI is an AI-based application designed to classify news articles as **Fake News** or **Real News** using Machine Learning and Deep Learning techniques.

The application provides two text-classification models — Logistic Regression and DistilBERT — allowing users to analyze news content and compare their predictions and confidence scores through an interactive web interface.

> **Disclaimer:** FactCheck AI provides machine-learning-based classification predictions. It does not independently verify the truth or authenticity of news articles.

---

## 🚀 Features

- **Fake News Detection:** Classifies news text as Fake or Real.
- **Dual ML Models:** Supports Logistic Regression and DistilBERT.
- **Confidence Scores:** Displays model confidence and available class probabilities.
- **Interactive Interface:** Simple and user-friendly React frontend.
- **REST API:** FastAPI backend for model inference.
- **Model Selection:** Allows users to choose between available classification models.
- **Health Monitoring:** Backend API provides model readiness information.

---

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Frontend | React.js, Vite, JavaScript |
| Backend | Python, FastAPI |
| Machine Learning | Scikit-learn, Logistic Regression |
| Deep Learning | PyTorch, DistilBERT |
| NLP | TF-IDF, Transformers |
| Model Storage | Joblib, Safetensors |
| API Testing | Swagger UI |
| Version Control | Git, GitHub |

---

## 🏗️ Project Architecture

```text
                 USER
                   |
                   v
          REACT FRONTEND
             (Vite)
                   |
                   v
             FASTAPI
              BACKEND
                   |
          +--------+--------+
          |                 |
          v                 v
    LOGISTIC REGRESSION   DISTILBERT
       (TF-IDF)          (PyTorch)
          |                 |
          v                 v
       PREDICTION        PREDICTION
          |                 |
          +--------+--------+
                   |
                   v
          RESULTS & CONFIDENCE
                   |
                   v
             USER INTERFACE
```

---

## 🧠 Machine Learning Models

FactCheck AI currently integrates two classification approaches.

### 1. Logistic Regression

- Uses TF-IDF for text feature extraction.
- Uses Logistic Regression for binary classification.
- Loads the trained pipeline using Joblib.
- Returns a predicted class and confidence score.

**Model file:** `backend/models/news_pipeline.joblib`

### 2. DistilBERT

- Uses a pretrained transformer-based architecture.
- Processes contextual information from news text.
- Uses PyTorch for model inference.
- Returns a classification prediction and class probabilities.

**Model directory:** `backend/models/factcheck_distilbert/`

### Model Comparison

Both models can produce different predictions for the same news article because they use different learning approaches.

A higher confidence score does not necessarily mean that a prediction is factually correct.

---

## 📁 Project Structure

```text
FactCheck-AI/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── distilbert_service.py
│   │
│   ├── data/
│   │   └── news.csv
│   │
│   ├── models/
│   │   ├── news_pipeline.joblib
│   │   └── factcheck_distilbert/
│   │       ├── config.json
│   │       ├── tokenizer.json
│   │       └── tokenizer_config.json
│   │
│   ├── clean_welfake.py
│   ├── split_welfake.py
│   ├── evaluate_models.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   └── App.jsx
│   ├── public/
│   ├── package.json
│   └── index.html
│
├── .gitignore
└── README.md
```

*Note: The repository also contains additional data-analysis and evaluation scripts. Large datasets and DistilBERT model weights are excluded from GitHub.*

---

## ⚙️ Installation and Setup

### Prerequisites

Install the following:

- Python 3.10 or later
- Node.js and npm
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/zafar-equbal/FactCheck-AI.git
```

Navigate into the project:

```bash
cd FactCheck-AI
```

### Step 2: Setup Backend

Open a terminal in the project directory.

Create a virtual environment:

```bash
python -m venv backend/.venv
```

Activate it on Windows:

```powershell
backend\.venv\Scripts\Activate.ps1
```

Navigate to the backend:

```bash
cd backend
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Backend will run at:

```text
http://127.0.0.1:8000
```

### Step 3: Setup Frontend

Open another terminal.

Navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the React development server:

```bash
npm run dev
```

Frontend will run at:

```text
http://localhost:5173
```

---

## 🔌 API Endpoints

The backend exposes the following REST API endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Check backend and model readiness |
| POST | `/api/analyze/news` | Analyze news using Logistic Regression |
| POST | `/api/analyze/news/distilbert` | Analyze news using DistilBERT |
| POST | `/api/analyze/media` | Media analysis endpoint (placeholder) |

### Example Request

**Endpoint:**

```text
POST /api/analyze/news
```

**Request Body:**

```json
{
  "text": "NASA successfully launched a new satellite to study Earth's climate."
}
```

### Example Response

```json
{
  "status": "ok",
  "model": "Logistic Regression",
  "prediction": "1",
  "confidence": 0.7954,
  "disclaimer": "Model prediction only—not proof of truth or authenticity. Verify important claims using reliable independent sources."
}
```

### Swagger API Documentation

After starting the backend, visit:

```text
http://127.0.0.1:8000/docs
```

---

## 📊 Dataset

The project uses the **WELFake dataset** for fake news classification.

The dataset was used for text preprocessing, model training, and evaluation.

Large dataset files are not included in this GitHub repository because of their size.

---

## ⚠️ Important Model Setup

Some large files are intentionally excluded from GitHub.

### Logistic Regression

The trained model is included:

```text
backend/models/news_pipeline.joblib
```

### DistilBERT

The trained model weights are approximately 255 MB and are not included in the repository.

To enable DistilBERT inference, place the following files in:

```text
backend/models/factcheck_distilbert/
```

Required files:

- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`

The configuration and tokenizer files are included in the repository. You must obtain the matching trained `model.safetensors` file separately and place it in the directory above.

---

## 🔮 Future Improvements

- News prediction history.
- Improved model comparison interface.
- Integration with trusted news sources and fact-checking APIs.
- Image and video authenticity analysis.
- User authentication.
- Cloud deployment.
- Improved model evaluation and generalization testing.

---

## 👨‍💻 Developer

**Zafar Equbal**

B.Tech Computer Science and Engineering

GitHub: [@zafar-equbal](https://github.com/zafar-equbal)

Project Repository: [FactCheck AI](https://github.com/zafar-equbal/FactCheck-AI)

---

## 📜 Disclaimer

FactCheck AI is an educational and experimental project developed for fake news classification.

Its predictions are generated using machine-learning models trained on a dataset. They should not be treated as definitive evidence that a news article is true or false.

Always verify important information through reliable and independent sources.   