# ai-experiments

Python web UI app for Gemini generation with:
- **Image mode**: upload a reference image + prompt to generate an image.
- **Audio mode**: provide a script to generate speech audio.
- **API key input directly in the UI**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).
