from src.stt.whisper.whisper_stt import WhisperArchitecture
from src.stt.dataset_for_train_loader import DatasetLoader
from src.stt.training_args.whisper_training_args import training_args
from src.config import settings


student = WhisperArchitecture(
    settings.student_model, 
    settings.device, 
    settings.torch_dtype
)

dataset_loader = DatasetLoader(settings, student.processor)
dataset = dataset_loader.load_dataset_for_train("train")

student.train(dataset, training_args)
