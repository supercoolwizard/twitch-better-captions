from src.services.bucket_manager_service import BucketManagerService
from src.adapters.huggingface import get_bucket_name
from src.config import settings

# relative_folder_name = str(settings.INPUT_DIR)[len(str(settings.BASE_DIR)):]
# print(relative_folder_name)

bms = BucketManagerService(get_bucket_name())
bms.create_bucket()
bms.upload_dir(settings.INPUT_DIR)
bms.download_dir(settings.INPUT_DIR)
