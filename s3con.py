import boto3
import os
class s3con:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            endpoint_url=os.getenv('ENDPOINT_URL'), 
            aws_access_key_id=os.getenv('KEY_ID'),
            aws_secret_access_key=os.getenv('ACCESS_KEY')
        )


    def download(self,  bucket_name : str, folder : str):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DIR = os.path.join(BASE_DIR, folder)

        contents = self.s3.list_objects_v2(Bucket=bucket_name)['Contents']
        keys = [contents[i]['Key'] for i in range(len(contents))]

        os.makedirs(DIR, exist_ok=True)   

        for key in keys:
            dst_path = os.path.join(DIR, key)
            self.s3.download_file(bucket_name, key, dst_path)
            print(f'скачан {key}')

        print('Скачан весь датасет')

    def upload_one(self, bucket_name : str, path : str):
        with open(path, "rb") as f:
                self.s3.upload_fileobj(f, bucket_name, path)
                print(f'загружен {path}')

    def upload(self,  bucket_name : str, folder : str):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DIR = os.path.join(BASE_DIR, folder)

        for file_name in os.listdir(DIR):
            file_path = os.path.join(DIR,file_name)
            with open(file_path, "rb") as f:
                self.s3.upload_fileobj(f, bucket_name, file_name)
                print(f'загружен {file_name}')

        print("Все файлы датасета загружены в s3")


if __name__ == '__main__':
    s3 = s3con()
    s3.download('vocab', 'vocab')
    s3.download('pairs-dataset', 'pairs')