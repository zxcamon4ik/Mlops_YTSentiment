# register model

import json
import mlflow
import logging
import os
import time

from dotenv import load_dotenv
from pathlib import Path

from src.model.config import get_mlflow_tracking_uri, get_model_alias, get_model_name, get_model_stage

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Set up MLflow tracking URI
mlflow.set_tracking_uri(get_mlflow_tracking_uri())

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def wait_until_ready(client: mlflow.tracking.MlflowClient, model_name: str, version: str) -> None:
    for _ in range(60):
        model_version = client.get_model_version(model_name, version)
        if getattr(model_version, "status", "READY") == "READY":
            return
        time.sleep(1)
    raise TimeoutError(f"Model {model_name} version {version} did not become READY")


def set_serving_target(client: mlflow.tracking.MlflowClient, model_name: str, version: str) -> None:
    model_alias = get_model_alias()
    model_stage = get_model_stage()

    if hasattr(client, "set_registered_model_alias") and model_alias:
        client.set_registered_model_alias(model_name, model_alias, version)
        logger.debug("Set alias %s for model %s version %s", model_alias, model_name, version)

    if model_stage:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=model_stage,
            archive_existing_versions=True,
        )
        logger.debug("Transitioned model %s version %s to %s", model_name, version, model_stage)


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        client = mlflow.tracking.MlflowClient()
        wait_until_ready(client, model_name, model_version.version)
        set_serving_target(client, model_name, model_version.version)
        
        logger.debug('Model %s version %s registered and set as serving target.', model_name, model_version.version)
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = get_model_name()
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()
