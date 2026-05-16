from huggingface_hub import create_bucket, delete_bucket
import subprocess

from src.utils.file_utils import FileUtils


class BucketManagerService:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.bucket_target_dir_name = "input"

    def create_bucket(self):
        create_bucket(self.bucket_name, 
                      private=True,
                      exist_ok=True)

    def delete_bucket(self):
        delete_bucket(self.bucket_name)

    def upload_dir(self, dir_to_upload):
        FileUtils.remove_ds_store(dir_to_upload)

        cmd = [
            "hf", "buckets",
            "sync", dir_to_upload,
            f"hf://buckets/{self.bucket_name}/{self.bucket_target_dir_name}"
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    def download_dir(self, dir_to_download):
        cmd = [
            "hf", "buckets",
            "sync", f"hf://buckets/{self.bucket_name}/{self.bucket_target_dir_name}",
            dir_to_download
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

