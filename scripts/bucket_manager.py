import argparse

from src.services.bucket_manager_service import BucketManagerService
from src.adapters.huggingface import get_bucket_name
from src.config import settings


def main():
    parser = argparse.ArgumentParser(
        description="create/upload/delete/download data on hf bucket"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("create_bucket")
    subparsers.add_parser("delete_bucket")

    upload_parser = subparsers.add_parser("upload_dir")
    upload_parser.add_argument("dir_name", choices=["input", "data"])
 
    download_parser = subparsers.add_parser("download_dir")
    download_parser.add_argument("dir_name", choices=["input", "data"])

    args = parser.parse_args()

    bms = BucketManagerService(get_bucket_name())

    match args.command:
        case "create_bucket":
            bms.create_bucket()
        case "delete_bucket":
            bms.delete_bucket()
        case "upload_dir":
            if args.dir_name == "input":
                bms.upload_dir(settings.INPUT_DIR)
            elif args.dir_name == "data":
                bms.upload_dir(settings.DATA_DIR)
        case "download_dir": 
            bms.download_dir(args.dir_name)

if __name__ == "__main__":
    main()
