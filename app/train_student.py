import typer
from enum import Enum
from typing import Annotated
from src.adapters.huggingface import *
from src.stt.providers.whisper_stt import WhisperArchitecture
from src.stt.dataset_loader import DatasetLoader
from src.config import settings

app = typer.Typer()

class SplitName(str, Enum):
    train = "train"
    test = "test"

def load_student():
    return WhisperArchitecture(settings.student_model, settings.device, settings.torch_dtype)


@app.command()
def train(
    split: Annotated[SplitName, typer.Option("--split")]
):
    dataset_loader = DatasetLoader(settings)
    dataset = dataset_loader.load_dataset(split.value)
    student = load_student()
    student.train(dataset)


if __name__ == "__main__":
    app()
