import boto3
import os
"""
Загрузить датасет в S3 хранилище
"""

bucket_name = "taiga" # изменить, если надо загрузить в другой бакет
dataset_folder = "dataset" # папка, в которой хранится датасет

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, dataset_folder)


s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('ENDPOINT_URL'),
    aws_access_key_id=os.getenv('KEY_ID'),
    aws_secret_access_key=os.getenv('ACCESS_KEY')
)

for file_name in os.listdir(DATASET_DIR):
    file_path = os.path.join(DATASET_DIR,file_name)
    with open(file_path, "rb") as f:
        s3.upload_fileobj(f, bucket_name, file_name)

print("Все файлы датасета загружены в s3")
