from pydantic import DirectoryPath
from pydantic_settings import BaseSettings
from pathlib import Path
import torch

class Settings(BaseSettings):
    # BASE_DIR: DirectoryPath = Path(__file__).resolve().parent.parent
    # DATA_DIR: DirectoryPath = BASE_DIR / "data"
    # INPUT_DIR: DirectoryPath = BASE_DIR / "input"
    # MODELS_DIR: DirectoryPath = BASE_DIR / "models"

    DATA_DIR: DirectoryPath = Path("data")
    INPUT_DIR: DirectoryPath = Path("input")
    MODELS_DIR: DirectoryPath = Path("models")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    student_model: str = "openai/whisper-tiny"
    teacher_model: str = "openai/whisper-large-v3"

    hf_repo_name: str = "whisper-tiny-lora-finetuned"
    hf_bucket_name: str = "broadcasts-dataset"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: torch.dtype = (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )


settings = Settings()
