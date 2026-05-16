from pydantic import DirectoryPath
from pydantic_settings import BaseSettings
from pathlib import Path
import torch

class Settings(BaseSettings):
    BASE_DIR: DirectoryPath = Path(__file__).resolve().parent.parent
    DATA_DIR: DirectoryPath = BASE_DIR / "data"
    INPUT_DIR: DirectoryPath = BASE_DIR / "input"
    MODELS_DIR: DirectoryPath = BASE_DIR / "models"

    student_model: str = "UsefulSensors/moonshine-tiny"
    teacher_model: str = "openai/whisper-tiny"

    hf_repo_name: str = "moonshine-lora-finetuned"
    hf_bucket_name: str = "broadcasts-dataset"

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


settings = Settings()
