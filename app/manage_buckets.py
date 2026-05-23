from typing import Literal

import typer

from src.services.bucket_manager_service import BucketManagerService
from src.adapters.huggingface import get_bucket_name
from src.config import settings

DirName = Literal["input", "data"]

app = typer.Typer()

bms = BucketManagerService(get_bucket_name())
path_map = {
    "input": settings.INPUT_DIR,
    "data": settings.DATA_DIR,
}

@app.command()
def create_bucket():
    bms.create_bucket()

@app.command()
def delete_bucket():
    bms.delete_bucket()

@app.command()
def upload_dir(dir_name: DirName):
    bms.upload_dir(path_map[dir_name])

@app.command()
def download_dir(dir_name: DirName):
    bms.download_dir(path_map[dir_name])


if __name__ == "__main__":
    app()
