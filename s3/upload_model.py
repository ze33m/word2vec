import boto3
import os
"""
Загрузить модель в S3
"""

bucket_name = "taiga" # изменить, если надо загрузить в другой бакет
model_folder = "model" # папка, в которой хранится модель
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, model_folder)


s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('ENDPOINT_URL'),
    aws_access_key_id=os.getenv('KEY_ID'),
    aws_secret_access_key=os.getenv('ACCESS_KEY')
)

for file_name in os.listdir(MODEL_DIR):
    file_path = os.path.join(MODEL_DIR, file_name)
    with open(file_path, "rb") as f:
        s3.upload_fileobj(f, bucket_name, file_name)
print("Модель загружена в s3")
