import mlflow
import pytest

from dotenv import load_dotenv
from pathlib import Path
import os
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.model.config import get_mlflow_tracking_uri, get_model_alias, get_model_name

pytestmark = pytest.mark.integration

if os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytestmark = [pytestmark, pytest.mark.skip(reason="Set RUN_INTEGRATION_TESTS=1 to run MLflow integration checks")]


mlflow.set_tracking_uri(get_mlflow_tracking_uri())


def test_model_accepts_raw_text_comments():
    model_name = get_model_name()
    model_alias = get_model_alias()

    try:
        model_uri = f"models:/{model_name}@{model_alias}"
        model = mlflow.pyfunc.load_model(model_uri)

        prediction = model.predict(["hi how are you"])

        assert len(prediction) == 1, "Output row count mismatch"

        print(f"Model '{model_name}' alias '{model_alias}' successfully processed raw text.")

    except Exception as e:
        pytest.fail(f"Model test failed with error: {e}")
