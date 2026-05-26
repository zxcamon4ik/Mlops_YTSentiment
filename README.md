# YouTube Sentiment Insights

An end-to-end machine learning prototype that analyzes the sentiment of YouTube
video comments and presents simple visual insights in a Chrome extension.

The project combines:

- a text classification pipeline based on TF-IDF features and LightGBM;
- DVC for reproducible pipeline stages and artifact tracking;
- MLflow for experiment logging and model registration;
- a Flask API for inference and chart generation;
- a Manifest V3 Chrome extension that collects YouTube comments and displays
  analysis results.

This repository is a prototype and learning project. It demonstrates the full
path from experimentation to a user-facing integration, while still having
several production-readiness items documented below.

## What It Does

When the Chrome extension is opened on a YouTube video page, it:

1. Extracts the YouTube video ID from the active browser tab.
2. Requests up to 500 top-level comments using the YouTube Data API v3.
3. Sends comment text and timestamps to the Flask service.
4. Uses the trained sentiment model to classify each comment as:
   - `-1`: negative
   - `0`: neutral
   - `1`: positive
5. Displays summary metrics, a sentiment pie chart, a monthly sentiment trend
   graph, a word cloud, and the first 25 classified comments.

## Architecture

```text
YouTube Data API
       |
       v
Chrome Extension (plugin-frontend/)
       |
       | HTTP JSON requests
       v
Flask API (flask_api/main.py)
       |
       | loads model and vectorizer
       v
MLflow Model Registry + local TF-IDF artifact

Training workflow:

data/reddit.csv
       |
       v
DVC: ingestion -> preprocessing -> model building -> evaluation -> registration
       |              |                 |                 |
       v              v                 v                 v
data/raw/     data/interim/       model/vectorizer     MLflow run/model
```

## Model Pipeline

The DVC pipeline is defined in `dvc.yaml`, with model parameters in
`params.yaml`.

| Stage | Script | Purpose | Output |
| --- | --- | --- | --- |
| `data_ingestion` | `src/data/data_ingestion.py` | Removes missing/duplicate/empty comments and performs an 80/20 train-test split. | `data/raw/` |
| `data_preprocessing` | `src/data/data_preprocessing.py` | Lowercases, cleans, removes selected stop words, and lemmatizes comments. | `data/interim/` |
| `model_building` | `src/model/model_building.py` | Fits TF-IDF features and trains a multiclass LightGBM classifier. | `lgbm_model.pkl`, `tfidf_vectorizer.pkl` |
| `model_evaluation` | `src/model/model_evaluation.py` | Evaluates the test split, logs artifacts and metrics to MLflow, and records run information. | `experiment_info.json` |
| `model_registration` | `src/model/register_model.py` | Registers the logged MLflow model and transitions it to staging. | Registered model version |

### Current Parameters

```yaml
data_ingestion:
  test_size: 0.20

model_building:
  ngram_range: [1, 3]
  max_features: 1000
  learning_rate: 0.09
  max_depth: 20
  n_estimators: 367
```

### Dataset

The tracked training source is `data/reddit.csv`, with columns:

| Column | Meaning |
| --- | --- |
| `clean_comment` | Input comment text |
| `category` | Sentiment label: `-1`, `0`, or `1` |

In the current dataset:

| Item | Count |
| --- | ---: |
| Source rows | 37,249 |
| Rows after initial ingestion cleanup | 36,793 |
| Negative (`-1`) comments | 8,250 |
| Neutral (`0`) comments | 12,772 |
| Positive (`1`) comments | 15,771 |

Although the user interface analyzes YouTube comments, the current training
source is named as Reddit data. Evaluation on a representative YouTube
validation dataset is a recommended next step before making quality claims.

## Repository Layout

```text
.
|-- data/
|   |-- reddit.csv                 # Versioned source dataset
|   |-- raw/                       # DVC/generated train and test split
|   `-- interim/                   # DVC/generated processed datasets
|-- flask_api/
|   |-- main.py                    # Flask inference and visualization API
|   `-- test.py                    # Local experimentation script
|-- notebooks/                     # EDA and model experiment notebooks
|-- plugin-frontend/
|   |-- manifest.json              # Chrome extension manifest/configuration
|   |-- popup.html                 # Popup UI
|   `-- popup.js                   # Comment retrieval and API integration
|-- scripts/                       # MLflow, promotion, and endpoint checks
|-- src/
|   |-- data/                      # Ingestion and preprocessing pipeline stages
|   `-- model/                     # Training, evaluation, and registration stages
|-- .env.example                  # Environment variable template
|-- dvc.yaml                      # Reproducible ML pipeline
|-- params.yaml                   # Training configuration
|-- environment.yml               # Conda environment definition
`-- requirements.txt              # Python dependencies
```

## Setup Guide

### Prerequisites

- Conda or Miniconda
- Python 3.12, if not using the provided Conda environment
- A reachable MLflow tracking server for evaluation, registration, and the
  committed Flask serving path
- AWS/S3 credentials if the configured MLflow artifact storage requires them
- Google Chrome or a Chromium-based browser for the extension
- A YouTube Data API v3 key for retrieving comments

### 1. Create the Python Environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate YTsentimentAnalyzer
```

Or install into an existing Python 3.12 environment:

```bash
python -m pip install -r requirements.txt
```

The preprocessing and serving code uses NLTK stop words and WordNet. If these
resources have not already been downloaded:

```bash
python -m nltk.downloader stopwords wordnet
```

### 2. Configure Environment Variables

Create a local `.env` file from the supplied example and fill in your actual
values:

```bash
cp .env.example .env
```

Expected variables:

```dotenv
SERVER_URL='http://your-mlflow-server:port'
AWS_ACCESS_KEY_ID='your-access-key'
AWS_SECRET_ACCESS_KEY='your-secret-key'
AWS_DEFAULT_REGION='eu-north-1'
```

Do not commit `.env` or real credentials. It is intentionally ignored.

### 3. Reproduce the Training Pipeline

Run the complete DVC workflow from the repository root:

```bash
dvc repro
```

The pipeline generates intermediate datasets, model artifacts, evaluation
metadata, and attempts to register a model in MLflow.

Useful commands:

```bash
dvc status
dvc dag
```

The present pipeline uses MLflow in its evaluation and registration stages, so
`SERVER_URL` and the relevant artifact-store credentials must be valid before
running the entire pipeline.

### 4. Run the Flask API

The committed API entrypoint is:

```bash
python flask_api/main.py
```

It listens on `http://localhost:5000` when started directly.

Important current behavior: `flask_api/main.py` loads MLflow model
`my_model`, version `4`, on application startup and loads
`./tfidf_vectorizer.pkl` locally. That model version and vectorizer must exist
and be compatible for the API to start and serve predictions.

### 5. Configure and Load the Chrome Extension

Edit the placeholder values in `plugin-frontend/manifest.json` for local use:

```json
{
  "API_KEY": "YOUR_YOUTUBE_DATA_API_V3_KEY",
  "API_URL": "http://localhost:5000"
}
```

Keep real API keys out of committed changes. For a production extension,
client-side storage of a YouTube API key should be replaced by a controlled
backend integration.

To load the extension in Chrome:

1. Open `chrome://extensions`.
2. Enable **Developer mode**.
3. Select **Load unpacked**.
4. Choose the `plugin-frontend/` directory.
5. Open a YouTube watch page and click the extension icon.

The extension currently recognizes URLs in the standard form:

```text
https://www.youtube.com/watch?v=<video_id>
```

## API Guide

### Health/Home Endpoint

```http
GET /
```

Returns a simple API welcome message.

### Predict Comments

```http
POST /predict
Content-Type: application/json
```

Request:

```json
{
  "comments": [
    "I love this video!",
    "This was disappointing."
  ]
}
```

Example response:

```json
[
  { "comment": "I love this video!", "sentiment": 1 },
  { "comment": "This was disappointing.", "sentiment": -1 }
]
```

### Predict Comments With Timestamps

This endpoint is used by the Chrome extension.

```http
POST /predict_with_timestamps
Content-Type: application/json
```

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

Example response:

```json
[
  {
    "comment": "Fantastic explanation.",
    "sentiment": "1",
    "timestamp": "2026-01-10T12:00:00Z"
  }
]
```

### Generate Sentiment Chart

```http
POST /generate_chart
Content-Type: application/json
```

Request:

```json
{
  "sentiment_counts": {
    "1": 10,
    "0": 4,
    "-1": 2
  }
}
```

Returns a PNG pie chart.

### Generate Word Cloud

```http
POST /generate_wordcloud
Content-Type: application/json
```

Request:

```json
{
  "comments": ["Great tutorial", "Very clear explanation"]
}
```

Returns a PNG word cloud.

### Generate Sentiment Trend Graph

```http
POST /generate_trend_graph
Content-Type: application/json
```

Request:

```json
{
  "sentiment_data": [
    { "timestamp": "2026-01-10T12:00:00Z", "sentiment": 1 },
    { "timestamp": "2026-02-10T12:00:00Z", "sentiment": 0 }
  ]
}
```

Returns a PNG graph of monthly sentiment percentages.

## MLflow Workflow

The pipeline currently performs the following MLflow actions:

1. `src/model/model_evaluation.py` logs the trained LightGBM estimator,
   inferred schema, vectorizer artifact, classification metrics, and confusion
   matrix.
2. It writes `experiment_info.json`, linking the local pipeline to an MLflow
   run artifact.
3. `src/model/register_model.py` registers the model as `my_model` and moves
   the new version into the `Staging` stage.

The helper scripts under `scripts/` were written against the model name
`yt_chrome_plugin_model`, while registration and serving currently use
`my_model`. Align these values before relying on automated promotion or
registry checks.

## Tests and Checks

The `scripts/` directory contains integration-oriented checks:

| Script | Purpose |
| --- | --- |
| `scripts/test_flask_api.py` | Sends requests to an API already running on localhost. |
| `scripts/test_load_model.py` | Loads a staging model from MLflow. |
| `scripts/test_model_signature.py` | Checks registry model input compatibility with the vectorizer. |
| `scripts/test_model_performance.py` | Checks minimum metrics on processed holdout data. |
| `scripts/promote_model.py` | Promotes a staging registry model to production. |

These scripts require additional preparation:

- install `pytest`, because it is not currently listed in
  `requirements.txt`;
- configure a reachable MLflow server in `.env`;
- align the MLflow model name used by registration, API serving, tests, and
  promotion;
- start the Flask API before executing the endpoint tests.

Example after those prerequisites are satisfied:

```bash
python -m pip install pytest
python -m pytest scripts/test_flask_api.py
```

## Git and Artifact Tracking

This repository uses an allowlist-style root `.gitignore`: it ignores
everything first with `**`, then explicitly permits known source and
configuration paths. The `data/.gitignore` file similarly allows only
`reddit.csv` and its own ignore file within `data/`.

Consequences for contributors:

- generated datasets, model pickle files, plots, MLflow runs, local `.env`
  files, and caches remain untracked by default;
- a newly added source directory or top-level documentation file will also be
  ignored until it is explicitly allowed in `.gitignore`;
- use `git status --ignored` or `git check-ignore -v <path>` when a new file
  does not appear in Git status;
- update `.gitignore` deliberately when adding new versioned project
  components.

Currently versioned project areas include:

```text
.env.example
.dvc/config
README.md
data/reddit.csv
dvc.lock
dvc.yaml
environment.yml
flask_api/
notebooks/
params.yaml
plugin-frontend/
requirements.txt
scripts/
setup.py
src/
```

## Prototype Limitations and Next Improvements

This prototype demonstrates the intended application workflow, but the
following items should be addressed before deployment:

1. Align the registered, promoted, tested, and served MLflow model identity.
   The current code uses both `my_model` and `yt_chrome_plugin_model`, and the
   API pins a specific model version.
2. Store preprocessing/vectorization together with the model, rather than
   loading a registry model with an independently managed local vectorizer.
3. Make pipeline stage exceptions fail the DVC command with a nonzero exit
   status.
4. Add `data/reddit.csv` as an explicit dependency of the DVC ingestion stage
   so dataset updates trigger pipeline reproduction.
5. Replace rendering of fetched comment content through `innerHTML` in the
   extension with safe DOM/text rendering.
6. Remove Flask debug mode for deployment and restrict CORS to intended
   extension/application origins.
7. Add unit tests and CI checks that do not require a live remote MLflow
   service.
8. Validate the trained classifier on YouTube-domain comments and publish
   meaningful performance results.

## Technology Stack

| Area | Tools |
| --- | --- |
| Language | Python 3.12, JavaScript |
| Data and ML | pandas, scikit-learn, LightGBM, NLTK |
| Visualization | matplotlib, seaborn, WordCloud |
| Experiment tracking and registry | MLflow |
| Pipeline and artifacts | DVC, optional S3-backed storage |
| Backend | Flask, Flask-CORS |
| Client | Chrome Extension Manifest V3, YouTube Data API v3 |

## Intended Scope

YouTube Sentiment Insights is best considered an end-to-end ML application
prototype: it connects research, pipeline tooling, a registry-backed model
service, and a browser interface in one repository. It is suitable for
demonstration, experimentation, and continued engineering development; the
limitations section identifies the main work needed to move toward a
production-grade service.
