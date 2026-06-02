import os
from dataclasses import dataclass


DEFAULT_MODEL_NAME = "youtube_sentiment_model"
DEFAULT_MODEL_ALIAS = "champion"
DEFAULT_MODEL_STAGE = "Production"
DEFAULT_MODEL_ARTIFACT_PATH = "model"
DEFAULT_PIPELINE_PATH = "sentiment_pipeline.pkl"
DEFAULT_MLFLOW_TRACKING_URI = "http://localhost:5001"


def get_mlflow_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)


def get_model_name() -> str:
    return os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)


def get_model_alias() -> str:
    return os.getenv("MODEL_ALIAS", DEFAULT_MODEL_ALIAS)


def get_model_stage() -> str:
    return os.getenv("MODEL_STAGE", DEFAULT_MODEL_STAGE)


def get_model_version() -> str | None:
    return os.getenv("MODEL_VERSION") or None


@dataclass(frozen=True)
class ModelTarget:
    model_name: str
    tracking_uri: str
    model_version: str | None = None
    model_alias: str | None = DEFAULT_MODEL_ALIAS
    model_stage: str | None = DEFAULT_MODEL_STAGE

    @classmethod
    def from_env(cls) -> "ModelTarget":
        return cls(
            model_name=get_model_name(),
            tracking_uri=get_mlflow_tracking_uri(),
            model_version=get_model_version(),
            model_alias=os.getenv("MODEL_ALIAS", DEFAULT_MODEL_ALIAS) or None,
            model_stage=os.getenv("MODEL_STAGE", DEFAULT_MODEL_STAGE) or None,
        )

    def candidate_uris(self) -> list[tuple[str, str, str]]:
        if self.model_version:
            return [
                (
                    f"models:/{self.model_name}/{self.model_version}",
                    "version",
                    self.model_version,
                )
            ]

        candidates: list[tuple[str, str, str]] = []
        if self.model_alias:
            candidates.append(
                (
                    f"models:/{self.model_name}@{self.model_alias}",
                    "alias",
                    self.model_alias,
                )
            )
        if self.model_stage:
            candidates.append(
                (
                    f"models:/{self.model_name}/{self.model_stage}",
                    "stage",
                    self.model_stage,
                )
            )
        return candidates
