from src.stt.dataset_for_train_loader import DatasetLoader
from src.config import settings
from src.stt.whisper.whisper_stt import WhisperArchitecture
import soundfile as sf


whisper = WhisperArchitecture(settings.student_model, settings.device, settings.torch_dtype)

dataset_loader = DatasetLoader(settings, whisper.processor)

# dummy = "input/ttt.mp3"
# audio, sr = sf.read(dummy)
# print(audio)
dataset_for_train = dataset_loader.load_dataset_for_train("train")

print(dataset_for_train)
