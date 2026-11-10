import { useEffect, useState } from 'react';
import {
  ShieldCheck, Search, FileText, Image, Video,
  Activity, AlertTriangle, LoaderCircle
} from 'lucide-react';

const API = 'http://127.0.0.1:8000';

export default function App() {
  const [tab, setTab] = useState('news');
  const [text, setText] = useState('');
  const [modelType, setModelType] = useState('distilbert');

  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/health`)
      .then(r => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: 'offline' }));
  }, []);

  async function analyze(e) {
    e.preventDefault();

    setError('');
    setResult(null);
    setLoading(true);

    const endpoint =
      modelType === 'distilbert'
        ? '/api/analyze/news/distilbert'
        : '/api/analyze/news';

    try {
      const response = await fetch(`${API}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Analysis failed');
      }

      setResult(data);

    } catch (err) {
      setError(err.message || 'Cannot connect to backend.');
    } finally {
      setLoading(false);
    }
  }

  const prediction = result
    ? Number(result.prediction) === 0
      ? 'FAKE NEWS'
      : Number(result.prediction) === 1
        ? 'REAL NEWS'
        : result.prediction
    : '';

  return (
    <div className="layout">

      <aside className="sidebar">

        <div className="brand">
          <div className="logo"><ShieldCheck /></div>
          <div>
            <b>FactCheck <i>AI</i></b>
            <small>TRUST, WITH EVIDENCE</small>
          </div>
        </div>

        <p className="nav-label">WORKSPACE</p>

        <button
          className={tab === 'news' ? 'nav active' : 'nav'}
          onClick={() => setTab('news')}
        >
          <FileText /> News analysis
        </button>

        <button
          className={tab === 'image' ? 'nav active' : 'nav'}
          onClick={() => setTab('image')}
        >
          <Image /> Image check <small>SOON</small>
        </button>

        <button
          className={tab === 'video' ? 'nav active' : 'nav'}
          onClick={() => setTab('video')}
        >
          <Video /> Deepfake video <small>SOON</small>
        </button>

        <div className="sidebar-foot">
          <span className={health?.status === 'ok' ? 'online' : 'offline'} />
          <div>
            <b>API status</b>
            <small>
              {health?.status === 'ok'
                ? 'Backend connected'
                : health?.status === 'offline'
                  ? 'Backend offline'
                  : 'Checking…'}
            </small>
          </div>
        </div>

      </aside>

      <main>

        <header>
          <span>
            Workspace　/　<b>News analysis</b>
          </span>
          <label>● AI DETECTION WORKSPACE</label>
        </header>

        <section className="hero">

          <div className="eyebrow">
            INTELLIGENCE FOR THE INFORMATION AGE
          </div>

          <h1>
            Look closer.<br />
            <em>Know better.</em>
          </h1>

          <p>
            Analyze suspicious news with AI-assisted signals.
            Review the model prediction, understand its limitations,
            and investigate further.
          </p>

          <div className="hero-foot">
            <span><ShieldCheck /> Evidence-aware analysis</span>
            <span><Activity /> Machine-learning powered</span>
          </div>

          <div className="hero-mark">
            <ShieldCheck size={58} />
            <b>FC</b>
          </div>

        </section>

        <section className="grid">

          <div className="panel">

            <div className="panel-head">
              <div>
                <small>01 / INPUT</small>
                <h2>Analyze content</h2>
              </div>
              <span className="chip">● TEXT MODEL</span>
            </div>

            {tab === 'news' ? (

              <form onSubmit={analyze}>

                <label htmlFor="model">
                  SELECT AI MODEL
                </label>

                <select
                  id="model"
                  value={modelType}
                  onChange={e => {
                    setModelType(e.target.value);
                    setResult(null);
                    setError('');
                  }}
                >
                  <option value="distilbert">
                    DistilBERT
                  </option>

                  <option value="logistic">
                    Logistic Regression
                  </option>
                </select>

                <label htmlFor="news">
                  NEWS TEXT OR CLAIM
                </label>

                <textarea
                  id="news"
                  value={text}
                  onChange={e => setText(e.target.value)}
                  minLength={20}
                  maxLength={20000}
                  placeholder="Paste a headline and article text to analyze… (minimum 20 characters)"
                  required
                />

                <div className="count">
                  {text.length.toLocaleString()} / 20,000 characters
                </div>

                <button
                  className="submit"
                  disabled={loading || text.trim().length < 20}
                >
                  {loading ? (
                    <>
                      <LoaderCircle className="spin" />
                      Analyzing…
                    </>
                  ) : (
                    <>
                      <Search />
                      Analyze news ↗
                    </>
                  )}
                </button>

              </form>

            ) : (

              <div className="placeholder">
                <div className="placeholder-icon">
                  {tab === 'image'
                    ? <Image size={28} />
                    : <Video size={28} />}
                </div>

                <h3>
                  {tab === 'image'
                    ? 'Image deepfake detection'
                    : 'Video deepfake detection'}
                </h3>

                <p>
                  This module is planned for the next phase.
                  Media preprocessing and model training still need
                  to be implemented.
                </p>

                <span>NOT IMPLEMENTED YET</span>
              </div>

            )}

            {error && (
              <div className="message error">
                <AlertTriangle />
                <div>
                  <b>Could not complete analysis</b>
                  <p>{error}</p>
                </div>
              </div>
            )}

            {result && (
              <div className="message result">
                <AlertTriangle />

                <div>
                  <small>MODEL PREDICTION</small>

                  <h3>{prediction}</h3>

                  <p>
                    Model: {result.model}
                  </p>

                  <p>
                    Confidence: {(result.confidence * 100).toFixed(2)}%
                  </p>

                  {result.fake_probability !== undefined && (
                    <>
                      <p>
                        Fake probability: {
                          (result.fake_probability * 100).toFixed(2)
                        }%
                      </p>

                      <p>
                        Real probability: {
                          (result.real_probability * 100).toFixed(2)
                        }%
                      </p>
                    </>
                  )}

                  <p>{result.disclaimer}</p>
                </div>
              </div>
            )}

          </div>

          <div className="right">

            <div className="panel">
              <small>02 / SYSTEM STATUS</small>
              <h2>Model readiness</h2>

              <div className="status-row">
                <FileText />
                <div>
                  <b>Logistic Regression</b>
                  <small>
                    {health?.news_model_ready
                      ? 'Saved model detected'
                      : 'Waiting for training'}
                  </small>
                </div>
                <span>
                  {health?.news_model_ready ? 'READY' : 'SETUP'}
                </span>
              </div>

              <div className="status-row">
                <Activity />
                <div>
                  <b>DistilBERT</b>
                  <small>
                    {health?.distilbert_model_ready
                      ? 'Trained model connected'
                      : 'Model not connected'}
                  </small>
                </div>
                <span>
                  {health?.distilbert_model_ready ? 'READY' : 'SETUP'}
                </span>
              </div>

              <div className="status-row">
                <Image />
                <div>
                  <b>Image deepfake model</b>
                  <small>Model not connected</small>
                </div>
                <span>PLANNED</span>
              </div>

              <div className="status-row">
                <Video />
                <div>
                  <b>Video analysis</b>
                  <small>Frame pipeline not built</small>
                </div>
                <span>PLANNED</span>
              </div>

            </div>

            <div className="warning">
              <AlertTriangle />
              <div>
                <b>AI is a signal, not a verdict.</b>
                <p>
                  Predictions can be wrong. Check original sources
                  and trusted reporting before sharing important claims.
                </p>
              </div>
            </div>

            <div className="panel">
              <small>YOUR NEXT MILESTONE</small>
              <h3>Build evidence-based verification</h3>
              <p className="muted">
                Combine AI classification with reliable evidence
                retrieval and independent source verification.
              </p>

              <div className="bar">
                <span
                  style={{
                    width: health?.distilbert_model_ready ? '55%' : '38%'
                  }}
                />
              </div>
            </div>

          </div>

        </section>

        <footer>
          <b>FACTCHECK AI</b>
          <span>Built for research, transparency, and responsible AI.</span>
          <span>2026 · v0.2</span>
        </footer>

      </main>

    </div>
  );
}