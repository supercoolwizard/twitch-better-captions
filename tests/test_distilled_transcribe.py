from src.services.transcription_service import TranscriptionService
from src.stt.providers.whisper_stt import WhisperArchitecture
from src.stt.dataset_loader import DatasetLoader
from src.config import settings
from src.adapters.huggingface import *

student_distilled = WhisperArchitecture(
    get_repo_name(), 
    settings.device, 
    settings.torch_dtype
)

dataset_loader = DatasetLoader(settings)
service = TranscriptionService(dataset_loader)

text = service.run(student_distilled, "student_distilled", "test")
print(text)

