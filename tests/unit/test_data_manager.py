from src.utils.data_manager import *
from src.config import settings

dm = DataManager(".mp3", settings.DATA_DIR, settings.INPUT_DIR)
# print(dm.train_test_splitter())
dm.train_test_splitter()
