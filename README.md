# Spam Classifier

A machine-learning web app that classifies SMS/email messages as **Spam** or **Not Spam**.
It uses a TF-IDF vectorizer + trained scikit-learn model, exposed through two interfaces:

- `app.py` — a **Flask** web app (HTML/CSS/JS frontend in `templates/` and `static/`)
- `hf.py` — a **Streamlit** app

## Project structure

```
project/
├── app.py                 # Flask web app
├── hf.py                  # Streamlit app
├── model1.pkl             # trained classifier
├── vectorizer1.pkl        # fitted TF-IDF vectorizer
├── spam_classifier_training.ipynb  # training notebook
├── templates/index.html   # Flask frontend
├── static/                # CSS + JS for the Flask frontend
├── requirements.txt       # Python dependencies
├── Procfile               # for Render deployment (Flask)
└── runtime.txt            # Python version for Render
```

## Run locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt   # ignore the gunicorn error on Windows
python app.py                     # Flask  -> http://localhost:5000
# or
streamlit run hf.py               # Streamlit -> http://localhost:8501
```

## Deploy live (free)

### Option A — Streamlit Community Cloud (easiest)
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. "New app" -> pick this repo, set **Main file path** to `hf.py`, Deploy.

### Option B — Render (Flask app)
1. Push this repo to GitHub.
2. Go to https://render.com -> "New" -> "Web Service" -> connect the repo.
3. Build command: `pip install -r requirements.txt`
   Start command: `gunicorn app:app`
4. Render reads `runtime.txt` for the Python version.

## Model artifacts
`model1.pkl` and `vectorizer1.pkl` are required at runtime and are committed to the repo
so the deployed app works without retraining.
