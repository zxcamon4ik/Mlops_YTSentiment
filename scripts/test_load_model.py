import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient


from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.model.config import get_mlflow_tracking_uri, get_model_alias, get_model_name

pytestmark = pytest.mark.integration

if os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytestmark = [pytestmark, pytest.mark.skip(reason="Set RUN_INTEGRATION_TESTS=1 to run MLflow integration checks")]


mlflow.set_tracking_uri(get_mlflow_tracking_uri())


def test_load_serving_model():
    model_name = get_model_name()
    model_alias = get_model_alias()
    client = MlflowClient()
    
    try:
        model_uri = f"models:/{model_name}@{model_alias}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Ensure the model loads successfully
        assert model is not None, "Model failed to load"
        print(f"Model '{model_name}' loaded successfully from alias '{model_alias}'.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
