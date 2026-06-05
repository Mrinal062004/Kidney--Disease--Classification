from Kidney import logger
from Kidney.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from Kidney.pipeline.stage02_prepare_base_model import PrepareBaseModelTrainingPipeline
STAGE_NAME="Data Ingestion stage"

try:
    logger.info(f">>>>> c {STAGE_NAME} started <<<<<<\n\nx========x")
    obj=DataIngestionTrainingPipeline()
    obj.main()
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME="Prepare Base Model"
try:
    logger.info(f"***************")
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    obj=PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>> stage {STAGE_NAME} completed<<<<\n\nx======x")
except Exception as e:
    logger.exception(e)
    raise e
