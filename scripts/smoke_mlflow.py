from guardrails.tracking import start
import mlflow

with start("smoke", "hello"):
    mlflow.log_param("ok", True)
    mlflow.log_metric("recall_at_fpr5", 1.0)
