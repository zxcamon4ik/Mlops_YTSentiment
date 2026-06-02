import os
import mlflow

from dotenv import load_dotenv
from pathlib import Path

from src.model.config import get_mlflow_tracking_uri, get_model_alias, get_model_name

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

def promote_model():
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())

    client = mlflow.MlflowClient()

    model_name = get_model_name()
    model_alias = get_model_alias()
    source_stage = os.getenv("SOURCE_STAGE", "Staging")
    target_stage = os.getenv("TARGET_STAGE", "Production")

    # Get the latest version in staging
    latest_version_staging = client.get_latest_versions(model_name, stages=[source_stage])[0].version

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=[target_stage])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage=target_stage
    )
    if hasattr(client, "set_registered_model_alias") and model_alias:
        client.set_registered_model_alias(model_name, model_alias, latest_version_staging)
    print(f"Model version {latest_version_staging} promoted to {target_stage}")

if __name__ == "__main__":
    promote_model()
