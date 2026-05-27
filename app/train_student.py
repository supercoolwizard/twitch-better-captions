import typer
from enum import Enum
from typing import Annotated
from src.adapters.huggingface import *
from src.stt.whisper.whisper_stt import WhisperArchitecture
from src.stt.dataset_for_train_loader import DatasetTrainLoader
from src.config import settings
from src.stt.training_args.whisper_training_args import training_args

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
    student = load_student()
    dataset_loader = DatasetTrainLoader(settings, student.processor)
    dataset = dataset_loader.load_dataset_for_train(split.value)
    student.train(dataset, training_args)


if __name__ == "__main__":
    app()
