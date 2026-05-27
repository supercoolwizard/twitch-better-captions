from transformers.data import data_collator
from src.stt.providers.whisper_stt import WhisperArchitecture
from src.stt.dataset_loader import DatasetLoader
from src.config import settings


student = WhisperArchitecture(
    settings.student_model, 
    settings.device, 
    settings.torch_dtype
)

dataset_loader = DatasetLoader(settings)
dataset = dataset_loader.load_dataset_for_train("train")

student.train(dataset)
