from src.config import settings
from src.utils.transcripts_manager import TranscriptsManager

tm = TranscriptsManager(settings.DATA_DIR)
print(tm.metadata_crawler("test", "Learning how to draw Pixelart Skulls and Heads [v2752070597]"))

