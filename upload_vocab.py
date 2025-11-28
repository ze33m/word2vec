import boto3
from tqdm import tqdm
import yaml
import os
bucket_name = "int-taiga"

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('ENDPOINT_URL'),
    aws_access_key_id=os.getenv('KEY_ID'),
    aws_secret_access_key=os.getenv('ACCESS_KEY')
)


with open('vocab.json', "r", encoding='utf-8') as f:
    s3.upload_fileobj(f, bucket_name,'vocab.json')
print("Все файлы датасета загружены в s3")
