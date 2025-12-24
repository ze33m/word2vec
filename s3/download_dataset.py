import boto3
import os
"""
Скачать датасет с S3 хранилища
"""

bucket_name = "int-taiga"  # изменить бакет, если нужен другой датасет
dataset_folder = "dataset" # папка, в которую скачается датасет
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, dataset_folder) 


s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('ENDPOINT_URL'), 
    aws_access_key_id=os.getenv('KEY_ID'),
    aws_secret_access_key=os.getenv('ACCESS_KEY')
)

contents = s3.list_objects_v2(Bucket=bucket_name)['Contents']
keys = [contents[i]['Key'] for i in range(len(contents))]

os.makedirs(DATASET_DIR, exist_ok=True)   

for key in keys:
    dst_path = os.path.join(DATASET_DIR, key)
    s3.download_file(bucket_name, key, dst_path)
    print(f'скачан {key}')

print('Скачан весь датасет')
