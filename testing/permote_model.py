import os
import mlflow

def promote_model():
    # Set up AWS MLflow tracking URI
    mlflow.set_tracking_uri("http://13.238.159.116:5000/")

    client = mlflow.MlflowClient()

    model_name = "plugin_model"
    # Get the latest version in staging
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version


    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()
    