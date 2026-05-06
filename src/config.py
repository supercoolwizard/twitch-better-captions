from pydantic import DirectoryPath
from pydantic_settings import BaseSettings
from pathlib import Path
import torch

class Settings(BaseSettings):
    DATA_DIR: DirectoryPath = Path("data")
    INPUT_DIR: DirectoryPath = Path("input")

    student_model: str = "UsefulSensors/moonshine-tiny"
    teacher_model: str ="openai/whisper-tiny"

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

settings = Settings()
