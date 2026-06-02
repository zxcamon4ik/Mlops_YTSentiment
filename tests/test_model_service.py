from types import SimpleNamespace

import api.model_service as model_service_module
from api.model_service import ModelService
from src.model.config import ModelTarget


class FakeMlflowModel:
    def predict(self, comments):
        return [1 for _ in comments]


def test_model_service_loads_alias_uri(monkeypatch):
    loaded_uris = []

    def fake_load_model(model_uri):
        loaded_uris.append(model_uri)
        return FakeMlflowModel()

    monkeypatch.setattr(model_service_module.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        model_service_module.mlflow,
        "pyfunc",
        SimpleNamespace(load_model=fake_load_model),
    )

    service = ModelService(
        ModelTarget(
            model_name="youtube_sentiment_model",
            tracking_uri="http://mlflow:5000",
            model_alias="champion",
            model_stage="Production",
        )
    )

    service.load_once()

    assert loaded_uris == ["models:/youtube_sentiment_model@champion"]
    assert service.health_payload()["model_loaded"] is True
    assert service.health_payload()["model_target"] == "champion"


def test_model_service_predict_uses_loaded_model(monkeypatch):
    monkeypatch.setattr(model_service_module.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        model_service_module.mlflow,
        "pyfunc",
        SimpleNamespace(load_model=lambda model_uri: FakeMlflowModel()),
    )
    service = ModelService(
        ModelTarget(
            model_name="youtube_sentiment_model",
            tracking_uri="http://mlflow:5000",
            model_alias="champion",
        )
    )

    assert service.predict(["one", "two"]) == [1, 1]
