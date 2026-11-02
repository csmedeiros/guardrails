"""Single entry point so every experiment logs to the same local MLflow store."""
import mlflow

TRACKING_URI = "sqlite:///mlflow/mlflow.db"


def start(experiment: str, run_name: str | None = None):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)
    return mlflow.start_run(run_name=run_name)
