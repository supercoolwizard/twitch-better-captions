from pydantic import DirectoryPath
from pydantic_settings import BaseSettings
from pathlib import Path
import torch

class Settings(BaseSettings):
    DATA_DIR: DirectoryPath = Path("data")
    INPUT_DIR: DirectoryPath = Path("input")
    MODELS_DIR: DirectoryPath = Path("models")

    student_model: str = "UsefulSensors/moonshine-tiny"
    teacher_model: str = "openai/whisper-tiny"

    hf_username: str = HfApi().whoami()["name"]
    hf_model_repo: str = f"{hf_username}/moonshine-lora-finetuned"


    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


settings = Settings()
