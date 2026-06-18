from Kidney.config.configuration import ConfigurationManager
from Kidney.components.model_training import Training
from Kidney import logger


STAGE_NAME="Model Training"
class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        training_config= config.get_training_config()
        training=Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(callback_list=[])



if __name__ =="__main__":
    try:
        logger.info(f"_______________________________________")
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj=ModelTrainingPipeline()
        obj.main()
        logger.info(f"stage {STAGE_NAME} completed <<<<< \n\n X---------------------")
    except Exception as e:
        logger.exception(e)
        raise e

