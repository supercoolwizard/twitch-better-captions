from src.config import settings
from src.stt.teacher import TeacherModel
from src.utils.transcripts_manager import TranscriptsManager
import torch
from unittest.mock import patch

teacher_model = "openai/whisper-tiny"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dummy_path = "data/test/audio/Learning how to draw Pixelart Skulls and Heads [v2752070597].mp3"
dummy_path = "data/test/audio/dummy.mp3"

tm = TeacherModel(teacher_model, device)
result = tm.local_hf_transcribe(dummy_path)
print(result)

