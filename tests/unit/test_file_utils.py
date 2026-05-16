from src.utils.file_utils import FileUtils
from src.config import settings

FileUtils.remove_ds_store(settings.INPUT_DIR)
