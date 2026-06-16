import os

normal = len(os.listdir("artifacts/data_ingestion/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Normal"))
tumor = len(os.listdir("artifacts/data_ingestion/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Tumor"))

print("Normal:", normal)
print("Tumor :", tumor)