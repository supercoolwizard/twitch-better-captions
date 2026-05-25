from huggingface_hub import create_bucket, delete_bucket
import subprocess

from src.utils.file_utils import FileUtils


class BucketManagerService:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name


    def create_bucket(self):
        create_bucket(self.bucket_name, 
                      private=True,
                      exist_ok=True)


    def delete_bucket(self):
        delete_bucket(self.bucket_name)


    # def clear_remote_dir(self, dir_name):
    #     remote_path = f"hf://buckets/{self.bucket_name}/{dir_name.name}"
    #
    #     cmd = [
    #         "hf", "buckets",
    #         "rm",
    #         remote_path
    #     ]
    #     subprocess.run(cmd, capture_output=True, text=True)


    def upload_dir(self, dir_name):
        FileUtils.remove_ds_store(dir_name)

        cmd = [
            "hf", "buckets",
            "sync", str(dir_name),
            f"hf://buckets/{self.bucket_name}/{dir_name.name}"
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)


    def download_dir(self, dir_name):
        cmd = [
            "hf", "buckets",
            "sync", f"hf://buckets/{self.bucket_name}/{dir_name}",
            dir_name
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

