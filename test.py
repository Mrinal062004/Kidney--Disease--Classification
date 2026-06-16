import dagshub
import mlflow

dagshub.init(
    repo_owner="Mrinal062004",
    repo_name="Kidney--Disease--Classification",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/Mrinal062004/Kidney--Disease--Classification.mlflow"
)

print("Tracking URI:", mlflow.get_tracking_uri())

mlflow.set_experiment("Default")

with mlflow.start_run():
    mlflow.log_param("test", 1)

print("SUCCESS")