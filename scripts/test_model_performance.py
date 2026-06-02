import pytest
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow


from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.model.config import get_mlflow_tracking_uri, get_model_alias, get_model_name

pytestmark = pytest.mark.integration

if os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytestmark = [pytestmark, pytest.mark.skip(reason="Set RUN_INTEGRATION_TESTS=1 to run MLflow integration checks")]


mlflow.set_tracking_uri(get_mlflow_tracking_uri())


def test_model_performance():
    model_name = get_model_name()
    model_alias = get_model_alias()
    holdout_data_path = "data/interim/test_processed.csv"

    try:
        model_uri = f"models:/{model_name}@{model_alias}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Load the holdout test data
        holdout_data = pd.read_csv(holdout_data_path)
        X_holdout_raw = holdout_data["clean_comment"].fillna("").astype(str)
        y_holdout = holdout_data["category"]

        y_pred_new = model.predict(X_holdout_raw.tolist())

        # Calculate performance metrics
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new, average='weighted', zero_division=1)
        recall_new = recall_score(y_holdout, y_pred_new, average='weighted', zero_division=1)
        f1_new = f1_score(y_holdout, y_pred_new, average='weighted', zero_division=1)


        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assert that the new model meets the performance thresholds
        assert accuracy_new >= expected_accuracy, f'Accuracy should be at least {expected_accuracy}, got {accuracy_new}'
        assert precision_new >= expected_precision, f'Precision should be at least {expected_precision}, got {precision_new}'
        assert recall_new >= expected_recall, f'Recall should be at least {expected_recall}, got {recall_new}'
        assert f1_new >= expected_f1, f'F1 score should be at least {expected_f1}, got {f1_new}'

        print(f"Performance test passed for model '{model_name}' alias '{model_alias}'")

    except Exception as e:
        pytest.fail(f"Model performance test failed with error: {e}")
