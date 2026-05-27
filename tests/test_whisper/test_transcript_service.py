from src.services.transcription_service import TranscriptionService
from src.stt.providers.whisper_stt import WhisperArchitecture
from src.stt.dataset_loader import DatasetLoader
from src.config import settings

student = WhisperArchitecture(
    settings.student_model, 
    settings.device, 
    settings.torch_dtype
)

teacher = WhisperArchitecture(
    settings.teacher_model, 
    settings.device, 
    settings.torch_dtype
)

dataset_loader = DatasetLoader(settings)
service = TranscriptionService(dataset_loader)

for role in ["student", "teacher"]:
    for split in ["train", "test"]:
        service.run(student, role, split)
        service.run(teacher, role, split)


