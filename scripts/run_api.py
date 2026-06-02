import logging
import os
import sys

import uvicorn

from api.model_service import ModelService


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def wait_for_registered_model() -> None:
    service = ModelService.from_env()
    retries = int(os.getenv("MODEL_WAIT_RETRIES", "60"))
    delay_seconds = float(os.getenv("MODEL_WAIT_SECONDS", "2"))

    if not service.load_with_retry(retries=retries, delay_seconds=delay_seconds):
        raise RuntimeError(
            f"Model {service.target.model_name} was not loadable from MLflow at "
            f"{service.target.tracking_uri}: {service.error}"
        )


def main() -> None:
    wait_for_registered_model()
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "5000"))
    uvicorn.run("api.main:app", host=host, port=port)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("API startup failed: %s", exc)
        sys.exit(1)
