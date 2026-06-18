import tensorflow as tf
from pathlib import Path
import dagshub
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from Kidney.entity.config_entity import EvaluationConfig
from Kidney.utils.common import read_yaml ,create_directories ,save_json

class Evaluation:
    def __init__(self,config:EvaluationConfig):
        self.config=config
    def valid_generator(self):
        datagenerator_kwargs=dict(
            rescale=1./255,
            validation_split=0.3
        )
        dataflow_kwargs=dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        valid_datagenerator =tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator=valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs

        )
    @staticmethod
    def load_model(path:Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    def evaluation(self):
        self.model=self.load_model(self.config.path_of_model)
        self.valid_generator()
        self.score=self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores={"loss":self.score[0],"accuracy":self.score[1]}
        save_json(path=Path("scores.json"),data=scores)


    # Ml flow code
    def log_into_mlflow(self):
        dagshub.init(
        repo_owner="Mrinal062004",
        repo_name="Kidney--Disease--Classification",
        mlflow=True
       )
  
        mlflow.set_tracking_uri("https://dagshub.com/Mrinal062004/Kidney--Disease--Classification.mlflow")

        print("Tracking URI:", mlflow.get_tracking_uri())
        mlflow.set_experiment("Kidney")

        with mlflow.start_run():
            mlflow.log_params({
            "image_size": str(self.config.params_image_size),
            "batch_size": self.config.params_batch_size
            })

            mlflow.log_metrics({
            "loss": self.score[0],
            "accuracy": self.score[1]
            })

            mlflow.keras.log_model(
            self.model,
            artifact_path="model",
            registered_model_name="VGG16Model"
            )