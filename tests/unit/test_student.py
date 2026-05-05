from src.config import settings
from src.stt.student import StudentModel
from src.utils.transcripts_manager import TranscriptsManager
import torch

student_model = "UsefulSensors/moonshine-tiny"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dummy_path = "data/test/audio/Learning how to draw Pixelart Skulls and Heads [v2752070597].mp3"
dummy_path = "data/test/audio/dummy.mp3"

sm = StudentModel(student_model, device)
print(sm.local_hf_transcribe(dummy_path))

