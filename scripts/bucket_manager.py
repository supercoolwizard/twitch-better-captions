import argparse

from src.services.bucket_manager_service import BucketManagerService
from src.adapters.huggingface import get_bucket_name
from src.config import settings


def main():
    parser = argparse.ArgumentParser(
        description="create/upload/delete/download data on hf bucket"
    )

    parser.add_argument(
        "--bucket_command",
        choices=["create_bucket", "delete_bucket"]
    )

    parser.add_argument(
        "--dir_command",
        choices=["upload_dir", "download_dir"]
    )

    args = parser.parse_args()

    if not args.bucket_command and not args.dir_command:
        parser.print_help()
        return

    bms = BucketManagerService(get_bucket_name())

    if args.bucket_command:
        match args.bucket_command:
            case "create_bucket":
                bms.create_bucket()
            case "delete_bucket":
                bms.delete_bucket()

    if args.dir_command:
        match args.dir_command:
            case "upload_dir":
                bms.upload_dir(settings.INPUT_DIR)
            case "download_dir": 
                bms.download_dir(settings.INPUT_DIR)

if __name__ == "__main__":
    main()
