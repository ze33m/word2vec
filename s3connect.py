import boto3
from tqdm import tqdm
import yaml
bucket_name = "taiga"
file_path = "preprocessed/taiga_preprocessed"

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

s3 = boto3.client(
    's3',
    endpoint_url='https://storage.yandexcloud.net',
    aws_access_key_id=config['s3']['id'],
    aws_secret_access_key=config['s3']['key']
)

with open(file_path, "rb") as f:
    s3.upload_fileobj(f, bucket_name, "preprocessed dataset")
print("Предобработанный датасет загружен")
