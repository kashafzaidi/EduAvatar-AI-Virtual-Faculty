# EduAvatar — AI Virtual Faculty

EduAvatar is a Streamlit-based GenAI learning assistant that lets students upload study PDFs, build a PDF knowledge base with embeddings + FAISS, learn topic-by-topic with an AI faculty, ask PDF-grounded questions, generate quizzes, and view learning analytics.

## Run locally

1. Install Python 3.12.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set your Gemini API key as `GEMINI_API_KEY` (environment variable or Streamlit secret).
5. Start the app:

```bash
streamlit run app.py
```

## Streamlit deployment

Use `app.py` as the main file. Add the following secret in the deployment platform's Secrets settings:

```toml
GEMINI_API_KEY = "your-api-key"
```

Do **not** commit the API key to GitHub.

## Main technologies

- Streamlit
- Google Gemini API
- Gemini Embeddings
- FAISS
- PyMuPDF
- scikit-learn
- pandas / NumPy / Matplotlib

## Project structure

- `app.py` — deployment entry point
- `app1.py` — main EduAvatar application
- `requirements.txt` — Python dependencies
- `runtime.txt` — Python 3.12 runtime
