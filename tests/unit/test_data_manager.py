from src.utils.data_manager import DataManager
from src.config import settings

dm = DataManager(".mp3", ".txt", settings.DATA_DIR, settings.INPUT_DIR)

dm.train_test_splitter()
dm.transcripts_files_creator()
dm.metadata_maker()
