import os
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

class S3Utils:
    def __init__(self, endpoint_url, access_key, secret_key, region_name='us-east-1'):
        self.s3 = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4'),
            region_name=region_name
        )

    def create_bucket_if_not_exists(self, bucket_name):
        try:
            self.s3.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                self.s3.create_bucket(Bucket=bucket_name)
                print(f"Bucket {bucket_name} created.")
            else:
                raise e

    def upload_file(self, file_path, bucket_name, object_name=None):
        if object_name is None:
            object_name = os.path.basename(file_path)
        try:
            self.s3.upload_file(file_path, bucket_name, object_name)
            return True
        except ClientError as e:
            print(f"Error uploading file: {e}")
            return False

    def download_file(self, bucket_name, object_name, file_path):
        try:
            self.s3.download_file(bucket_name, object_name, file_path)
            return True
        except ClientError as e:
            print(f"Error downloading file: {e}")
            return False

    def list_objects(self, bucket_name, prefix=''):
        try:
            response = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except ClientError as e:
            print(f"Error listing objects: {e}")
            return []

    def get_latest_object(self, bucket_name, prefix=''):
        try:
            response = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            if 'Contents' in response:
                # Sort by LastModified descending
                sorted_objs = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
                return sorted_objs[0]['Key']
            return None
        except ClientError as e:
            print(f"Error getting latest object: {e}")
            return None
