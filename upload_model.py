import boto3
from tqdm import tqdm
import yaml
import os
bucket_name = "taiga"
files_path = "model"

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('ENDPOINT_URL'),
    aws_access_key_id=os.getenv('KEY_ID'),
    aws_secret_access_key=os.getenv('ACCESS_KEY')
)

for file_name in os.listdir(files_path):
    file_path = files_path + '/' + file_name
    with open(file_path, "rb") as f:
        s3.upload_fileobj(f, bucket_name, file_name)
print("Модель загружена в s3")
