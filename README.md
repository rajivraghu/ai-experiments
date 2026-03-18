# ai-experiments

Python Streamlit app for Gemini-powered media workflows.

## Features

- **Image mode**: upload a reference image + prompt and generate a new image.
- **Audio mode**: enter a script and generate speech audio with Gemini TTS.
- **JSON Video Builder**:
  - paste or upload a JSON array of news/story objects
  - search and download high-resolution images from SearchApi.io using `image_search_keywords`
  - require downloaded images to have a dimension above 1280 pixels
  - send the downloaded image to Gemini image generation with the story `title` as the overlay prompt
  - convert each Telugu `summary` into speech audio using Gemini TTS
  - create one video per story from the generated image + audio
  - combine all story clips into a final MP4 or WebM reel

## Expected JSON shape

```json
[
  {
    "title": "Story title",
    "summary": "Telugu summary text",
    "image_search_keywords": "search keywords for SearchApi.io",
    "article_url": "https://example.com/article"
  }
]
```

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

Then open the local URL shown by Streamlit, typically `http://localhost:8501`.
