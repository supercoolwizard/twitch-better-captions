from pydantic import DirectoryPath
from pydantic_settings import BaseSettings
from pathlib import Path
import torch

class Settings(BaseSettings):
    BASE_DIR: DirectoryPath = Path(__file__).resolve().parent.parent
    DATA_DIR: DirectoryPath = BASE_DIR / "data"
    INPUT_DIR: DirectoryPath = BASE_DIR / "inputs"
    MODELS_DIR: DirectoryPath = BASE_DIR / "models"

    student_model: str = "UsefulSensors/moonshine-tiny"
    teacher_model: str = "openai/whisper-tiny"

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


settings = Settings()
