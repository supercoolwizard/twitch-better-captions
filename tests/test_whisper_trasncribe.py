from src.stt.providers.whisper_stt import WhisperArchitecture
from src.stt.dataset_loader import DatasetLoader
from src.config import settings

student = WhisperArchitecture(
    settings.student_model, 
    settings.device, 
    settings.torch_dtype
)

dataset_loader = DatasetLoader(settings)
dataset = dataset_loader.load_dataset("train")
inputs = dataset[0]["audio"]

# print(dataset)
# print(dataset[0]["teacher_path"])
# print(len(dataset))
print(student.transcribe(inputs))


