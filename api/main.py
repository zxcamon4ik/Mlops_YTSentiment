import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field
from wordcloud import WordCloud

from api.model_service import ModelService
from src.common.text_preprocessing import preprocess_comment, sentiment_stopwords


logger = logging.getLogger(__name__)
model_service = ModelService.from_env()


class CommentListInput(BaseModel):
    comments: List[str] = Field(default_factory=list)


class TimestampedComment(BaseModel):
    text: str
    timestamp: str


class TimestampedCommentInput(BaseModel):
    comments: List[TimestampedComment] = Field(default_factory=list)


class SentimentChartInput(BaseModel):
    sentiment_counts: Dict[str, int] = Field(default_factory=dict)


class WordCloudInput(BaseModel):
    comments: List[str] = Field(default_factory=list)


class SentimentTrendItem(BaseModel):
    timestamp: str
    sentiment: int


class TrendGraphInput(BaseModel):
    sentiment_data: List[SentimentTrendItem] = Field(default_factory=list)


def load_model_on_startup() -> None:
    if os.getenv("SKIP_MODEL_LOAD_ON_STARTUP", "").lower() in {"1", "true", "yes"}:
        return

    retries = int(os.getenv("MODEL_LOAD_RETRIES", "30"))
    delay_seconds = float(os.getenv("MODEL_LOAD_RETRY_SECONDS", "2"))
    model_service.load_with_retry(retries=retries, delay_seconds=delay_seconds)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model_on_startup()
    yield


app = FastAPI(
    title="YouTube Sentiment Insights API",
    description="FastAPI inference and visualization service for YouTube comment sentiment.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=PlainTextResponse)
def home() -> str:
    return "Welcome to YouTube Sentiment Insights API"


@app.get("/health")
def health() -> dict:
    return model_service.health_payload()


def error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def png_response(image_buffer: io.BytesIO) -> Response:
    return Response(content=image_buffer.getvalue(), media_type="image/png")


def predict_comments(comments: list[str]) -> list:
    try:
        return model_service.predict(comments)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise RuntimeError(f"Prediction failed: {exc}") from exc


@app.post("/predict")
def predict(payload: CommentListInput):
    comments = payload.comments
    if not comments:
        return error_response("No comments provided", 400)

    try:
        predictions = predict_comments(comments)
    except RuntimeError as exc:
        return error_response(str(exc), 503)

    return [
        {"comment": comment, "sentiment": sentiment}
        for comment, sentiment in zip(comments, predictions)
    ]


@app.post("/predict_with_timestamps")
def predict_with_timestamps(payload: TimestampedCommentInput):
    comments_data = payload.comments
    if not comments_data:
        return error_response("No comments provided", 400)

    comments = [item.text for item in comments_data]
    timestamps = [item.timestamp for item in comments_data]

    try:
        predictions = predict_comments(comments)
    except RuntimeError as exc:
        return error_response(str(exc), 503)

    sentiments = [str(prediction) for prediction in predictions]
    return [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
        for comment, sentiment, timestamp in zip(comments, sentiments, timestamps)
    ]


@app.post("/generate_chart")
def generate_chart(payload: SentimentChartInput):
    sentiment_counts = payload.sentiment_counts
    if not sentiment_counts:
        return error_response("No sentiment counts provided", 400)

    try:
        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0)),
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ["#36A2EB", "#C9CBCF", "#FF6384"]
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": "w"},
        )
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()
        return png_response(img_io)
    except Exception as exc:
        logger.exception("Chart generation failed")
        return error_response(f"Chart generation failed: {exc}", 500)


@app.post("/generate_wordcloud")
def generate_wordcloud(payload: WordCloudInput):
    comments = payload.comments
    if not comments:
        return error_response("No comments provided", 400)

    try:
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        text = " ".join(preprocessed_comments).strip() or "no comments"

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=sentiment_stopwords(),
            collocations=False,
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)
        return png_response(img_io)
    except Exception as exc:
        logger.exception("Word cloud generation failed")
        return error_response(f"Word cloud generation failed: {exc}", 500)


@app.post("/generate_trend_graph")
def generate_trend_graph(payload: TrendGraphInput):
    sentiment_data = [item.model_dump() for item in payload.sentiment_data]
    if not sentiment_data:
        return error_response("No sentiment data provided", 400)

    try:
        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["sentiment"] = df["sentiment"].astype(int)

        sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        monthly_counts = df.resample("ME")["sentiment"].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: "red", 0: "gray", 1: "green"}

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker="o",
                linestyle="-",
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value],
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()
        return png_response(img_io)
    except Exception as exc:
        logger.exception("Trend graph generation failed")
        return error_response(f"Trend graph generation failed: {exc}", 500)
