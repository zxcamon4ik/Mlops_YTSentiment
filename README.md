# YouTube Sentiment Insights

An end-to-end machine learning prototype that analyzes YouTube video comments
and presents sentiment summaries in a Chrome extension.

The project combines:

- a packaged sklearn inference pipeline with preprocessing, TF-IDF, and LightGBM;
- DVC for reproducible training stages;
- MLflow for experiment tracking and model registration;
- a FastAPI backend for inference and chart generation;
- a Manifest V3 Chrome extension in `plugin-frontend/`.

## What It Does

When the Chrome extension is opened on a YouTube video page, it:

1. Extracts the video ID from the active tab.
2. Requests up to 500 top-level comments using the YouTube Data API v3.
3. Sends comment text and timestamps to the backend.
4. Classifies comments as `-1` negative, `0` neutral, or `1` positive.
5. Displays metrics, a pie chart, a trend graph, a word cloud, and sampled
   classified comments.

## Architecture

```text
Chrome Extension (plugin-frontend/)
       |
       | HTTP JSON/PNG
       v
FastAPI API (api/main.py)
       |
       | loads MLflow model
       v
MLflow Model Registry

Training:

data/reddit.csv
       |
       v
DVC: ingestion -> preprocessing -> model building -> evaluation -> registration
```

The served MLflow model is a single pipeline artifact. The API no longer loads
`tfidf_vectorizer.pkl` or any separate preprocessing artifact.

## Unified Model Identity

The default MLflow model name is:

```dotenv
MODEL_NAME=youtube_sentiment_model
MODEL_ALIAS=champion
MODEL_STAGE=Production
```

Serving chooses the model target in this order:

1. `MODEL_VERSION`, when explicitly set.
2. `MODEL_ALIAS`, defaulting to `champion`.
3. `MODEL_STAGE`, defaulting to `Production`.

Registration sets the configured alias when supported by MLflow and also moves
the registered version to the configured stage as a fallback.

## Local Docker Run

From a clean clone, start the full local stack with one command:

```bash
docker compose up --build
```

Services:

| Service | URL |
| --- | --- |
| FastAPI API | `http://localhost:5000` |
| Swagger UI | `http://localhost:5000/docs` |
| OpenAPI schema | `http://localhost:5000/openapi.json` |
| MLflow UI | `http://localhost:5001` |

Compose starts:

- `mlflow`: local tracking server with a Docker volume for metadata/artifacts;
- `model-bootstrap`: waits for MLflow, downloads NLTK resources, runs `dvc repro`,
  logs the full pipeline, registers `youtube_sentiment_model`, and exits;
- `api`: waits for the registered model target and then starts Uvicorn on port
  `5000`.

No AWS or S3 credentials are required for the local Compose stack.

## Local Python Setup

```bash
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
```

Create a local environment file when running outside Docker:

```bash
cp .env.example .env
```

Default `.env.example` values:

```dotenv
MLFLOW_TRACKING_URI=http://localhost:5001
MODEL_NAME=youtube_sentiment_model
MODEL_ALIAS=champion
MODEL_STAGE=Production
API_HOST=0.0.0.0
API_PORT=5000
```

Run the training and registration pipeline:

```bash
dvc repro
```

Run the API directly:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 5000
```

## API

FastAPI provides Swagger at `/docs` and the OpenAPI schema at `/openapi.json`.

### `GET /`

Returns a plain text welcome message.

### `GET /health`

Returns model load status, model identity, serving target, model URI, and
MLflow tracking URI.

### `POST /predict`

Request:

```json
{
  "comments": ["I love this video!", "This was disappointing."]
}
```

Response:

```json
[
  { "comment": "I love this video!", "sentiment": 1 },
  { "comment": "This was disappointing.", "sentiment": -1 }
]
```

### `POST /predict_with_timestamps`

Used by the Chrome extension.

Request:

```json
{
  "comments": [
    {
      "text": "Fantastic explanation.",
      "timestamp": "2026-01-10T12:00:00Z"
    }
  ]
}
```

Response:

```json
[
  {
    "comment": "Fantastic explanation.",
    "sentiment": "1",
    "timestamp": "2026-01-10T12:00:00Z"
  }
]
```

### Image Endpoints

These return `image/png`:

- `POST /generate_chart` with `{"sentiment_counts": {"1": 10, "0": 4, "-1": 2}}`
- `POST /generate_wordcloud` with `{"comments": ["Great tutorial"]}`
- `POST /generate_trend_graph` with
  `{"sentiment_data": [{"timestamp": "2026-01-10", "sentiment": 1}]}`

## Chrome Extension

The extension still reads the backend URL from `plugin-frontend/manifest.json`:

```json
{
  "API_URL": "http://localhost:5000"
}
```

The backend route names and response shapes used by `popup.js` are preserved.
You still need to configure a YouTube Data API v3 key in the manifest for
comment retrieval.

## Tests

Unit tests do not require a live MLflow server:

```bash
python -m pytest
```

Integration-style checks under `scripts/` can be run separately after starting
the Docker stack:

```bash
RUN_INTEGRATION_TESTS=1 python -m pytest scripts
```

## Repository Layout

```text
api/                         FastAPI app and MLflow loading wrapper
src/common/                  Shared text preprocessing
src/data/                    DVC ingestion and preprocessing stages
src/model/                   Pipeline building, evaluation, registration
scripts/                     Bootstrap, API launcher, integration checks
plugin-frontend/             Chrome extension
data/reddit.csv              Versioned source dataset
dvc.yaml                     Reproducible training pipeline
docker-compose.yml           Local MLflow/bootstrap/API stack
```

## Notes

This is intentionally a bachelor-thesis-scale prototype. The implementation
keeps the serving path explicit: train a full pipeline, log real evaluation
metrics, register one model name, and serve the configured MLflow target.
