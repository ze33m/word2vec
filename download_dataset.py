import boto3
from tqdm import tqdm
import yaml
import os
bucket_name = "int-taiga"


s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('ENDPOINT_URL'), 
    aws_access_key_id=os.getenv('KEY_ID'),
    aws_secret_access_key=os.getenv('ACCESS_KEY')
)

contents = s3.list_objects_v2(Bucket=bucket_name)['Contents']
keys = [contents[i]['Key'] for i in range(len(contents))]

os.makedirs("dataset", exist_ok=True)   

for key in keys:
    s3.download_file(bucket_name, key, f'dataset/{key}')
    print(f'скачан {key}')

print('Скачан весь датасет')
