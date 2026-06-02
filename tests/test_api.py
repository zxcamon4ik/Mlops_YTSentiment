import os

os.environ["SKIP_MODEL_LOAD_ON_STARTUP"] = "true"

import api.main as api_main


class FakeModelService:
    def health_payload(self):
        return {
            "status": "loaded",
            "model_loaded": True,
            "model_name": "youtube_sentiment_model",
            "model_target_type": "alias",
            "model_target": "champion",
            "model_uri": "models:/youtube_sentiment_model@champion",
            "mlflow_tracking_uri": "http://mlflow:5000",
            "error": None,
        }

    def predict(self, comments):
        predictions = []
        for comment in comments:
            text = comment.lower()
            if "love" in text or "great" in text:
                predictions.append(1)
            elif "bad" in text or "worst" in text:
                predictions.append(-1)
            else:
                predictions.append(0)
        return predictions


def png_signature(response):
    assert response.media_type == "image/png"
    assert response.body.startswith(b"\x89PNG")


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(api_main, "model_service", FakeModelService())

    response = api_main.health()

    assert response["model_loaded"] is True
    assert response["model_name"] == "youtube_sentiment_model"


def test_predict(monkeypatch):
    monkeypatch.setattr(api_main, "model_service", FakeModelService())

    response = api_main.predict(
        api_main.CommentListInput(comments=["I love this", "This is the worst"])
    )

    assert response == [
        {"comment": "I love this", "sentiment": 1},
        {"comment": "This is the worst", "sentiment": -1},
    ]


def test_predict_with_timestamps(monkeypatch):
    monkeypatch.setattr(api_main, "model_service", FakeModelService())

    response = api_main.predict_with_timestamps(
        api_main.TimestampedCommentInput(
            comments=[
                api_main.TimestampedComment(
                    text="Great explanation",
                    timestamp="2026-01-10T12:00:00Z",
                ),
                api_main.TimestampedComment(
                    text="It was fine",
                    timestamp="2026-01-11T12:00:00Z",
                ),
            ]
        )
    )

    assert response == [
        {
            "comment": "Great explanation",
            "sentiment": "1",
            "timestamp": "2026-01-10T12:00:00Z",
        },
        {
            "comment": "It was fine",
            "sentiment": "0",
            "timestamp": "2026-01-11T12:00:00Z",
        },
    ]


def test_generate_chart_returns_png(monkeypatch):
    monkeypatch.setattr(api_main, "model_service", FakeModelService())

    response = api_main.generate_chart(
        api_main.SentimentChartInput(sentiment_counts={"1": 5, "0": 3, "-1": 2})
    )

    png_signature(response)


def test_generate_wordcloud_returns_png(monkeypatch):
    monkeypatch.setattr(api_main, "model_service", FakeModelService())

    response = api_main.generate_wordcloud(
        api_main.WordCloudInput(
            comments=["Love this tutorial", "Not so great", "Very clear"]
        )
    )

    png_signature(response)


def test_generate_trend_graph_returns_png(monkeypatch):
    monkeypatch.setattr(api_main, "model_service", FakeModelService())

    response = api_main.generate_trend_graph(
        api_main.TrendGraphInput(
            sentiment_data=[
                api_main.SentimentTrendItem(timestamp="2026-01-01", sentiment=1),
                api_main.SentimentTrendItem(timestamp="2026-01-02", sentiment=0),
                api_main.SentimentTrendItem(timestamp="2026-02-03", sentiment=-1),
            ]
        )
    )

    png_signature(response)
