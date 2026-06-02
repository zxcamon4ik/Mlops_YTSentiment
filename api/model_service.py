import logging
import time
from dataclasses import dataclass
from typing import Any

import mlflow

from src.model.config import ModelTarget


logger = logging.getLogger(__name__)


def prediction_list(predictions: Any) -> list:
    if hasattr(predictions, "tolist"):
        return predictions.tolist()
    return list(predictions)


@dataclass
class LoadedTarget:
    model_uri: str | None = None
    target_type: str | None = None
    target_value: str | None = None


class ModelService:
    def __init__(self, target: ModelTarget | None = None):
        self.target = target or ModelTarget.from_env()
        self.model = None
        self.loaded_target = LoadedTarget()
        self.status = "not_loaded"
        self.error: str | None = None

    @classmethod
    def from_env(cls) -> "ModelService":
        return cls(ModelTarget.from_env())

    def load_once(self) -> None:
        last_error: Exception | None = None
        mlflow.set_tracking_uri(self.target.tracking_uri)

        for model_uri, target_type, target_value in self.target.candidate_uris():
            try:
                self.model = mlflow.pyfunc.load_model(model_uri)
                self.loaded_target = LoadedTarget(model_uri, target_type, target_value)
                self.status = "loaded"
                self.error = None
                logger.info("Loaded MLflow model from %s", model_uri)
                return
            except Exception as exc:
                last_error = exc
                logger.warning("Could not load MLflow model from %s: %s", model_uri, exc)

        self.status = "failed"
        self.error = str(last_error) if last_error else "No model URI candidates configured"
        raise RuntimeError(self.error)

    def load_with_retry(self, retries: int = 30, delay_seconds: float = 2.0) -> bool:
        attempts = max(retries, 0) + 1
        for attempt in range(1, attempts + 1):
            try:
                self.load_once()
                return True
            except Exception as exc:
                self.error = str(exc)
                if attempt == attempts:
                    logger.error("Model loading failed after %s attempts: %s", attempts, exc)
                    return False
                logger.info(
                    "Waiting for model registry target %s/%s, attempt %s/%s",
                    self.target.model_name,
                    self.target.model_alias or self.target.model_stage or self.target.model_version,
                    attempt,
                    attempts,
                )
                time.sleep(delay_seconds)
        return False

    def predict(self, comments: list[str]) -> list:
        if self.model is None:
            self.load_once()
        return prediction_list(self.model.predict(comments))

    def configured_target(self) -> tuple[str | None, str | None]:
        if self.target.model_version:
            return "version", self.target.model_version
        if self.target.model_alias:
            return "alias", self.target.model_alias
        if self.target.model_stage:
            return "stage", self.target.model_stage
        return None, None

    def health_payload(self) -> dict:
        configured_type, configured_value = self.configured_target()
        return {
            "status": self.status,
            "model_loaded": self.model is not None,
            "model_name": self.target.model_name,
            "model_target_type": self.loaded_target.target_type or configured_type,
            "model_target": self.loaded_target.target_value or configured_value,
            "model_uri": self.loaded_target.model_uri,
            "mlflow_tracking_uri": self.target.tracking_uri,
            "error": self.error,
        }
