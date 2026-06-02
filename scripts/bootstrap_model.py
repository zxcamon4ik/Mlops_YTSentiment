import logging
import os
import subprocess
import sys
import time

import mlflow
from mlflow.tracking import MlflowClient

from src.common.text_preprocessing import download_nltk_resources
from src.model.config import get_mlflow_tracking_uri, get_model_name


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def wait_for_mlflow(tracking_uri: str, retries: int = 60, delay_seconds: float = 2.0) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    for attempt in range(1, retries + 1):
        try:
            client.search_experiments(max_results=1)
            logger.info("MLflow is reachable at %s", tracking_uri)
            return
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"MLflow was not reachable at {tracking_uri}: {exc}") from exc
            logger.info("Waiting for MLflow at %s, attempt %s/%s", tracking_uri, attempt, retries)
            time.sleep(delay_seconds)


def ensure_git_repo() -> None:
    if os.path.isdir(".git"):
        return

    logger.info("Initializing a local Git repository for DVC inside the container")
    subprocess.run(["git", "init"], check=True)
    subprocess.run(["git", "config", "user.email", "bootstrap@example.local"], check=True)
    subprocess.run(["git", "config", "user.name", "Model Bootstrap"], check=True)


def run_dvc_repro() -> None:
    logger.info("Running DVC pipeline to train and register %s", get_model_name())
    ensure_git_repo()
    subprocess.run(["dvc", "repro"], check=True)


def main() -> None:
    tracking_uri = get_mlflow_tracking_uri()
    wait_for_mlflow(tracking_uri)
    download_nltk_resources(os.getenv("NLTK_DATA"))
    run_dvc_repro()
    logger.info("Model bootstrap completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Model bootstrap failed: %s", exc)
        sys.exit(1)
