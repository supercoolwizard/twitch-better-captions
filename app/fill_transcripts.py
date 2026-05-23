import typer
from enum import Enum
from typing import Annotated
from src.config import settings
from src.services.transcription_service import TranscriptionService
from src.stt.dataset_loader import DatasetLoader
from src.stt.providers.whisper_stt import WhisperArchitecture
from src.adapters.huggingface import *
from src.stt.providers.whisper_stt import WhisperArchitecture

app = typer.Typer()

class Roles(str, Enum):
    student = "student"
    teacher = "teacher"
    student_distilled = "student_distilled"

class SplitName(str, Enum):
    train = "train"
    test = "test"

def get_instance(role):
    if role == Roles.student:
        return WhisperArchitecture(settings.student_model, settings.device, settings.torch_dtype)
    if role == Roles.teacher:
        return WhisperArchitecture(settings.teacher_model, settings.device, settings.torch_dtype)
    if role == Roles.student_distilled:
        return WhisperArchitecture(get_repo_name(), settings.device, settings.torch_dtype)


@app.command()
def transcribe(
    role: Annotated[Roles, typer.Option("--role")],
    split: Annotated[SplitName, typer.Option("--split")]
):
    stt_instance = get_instance(role)
    dataset_loader = DatasetLoader(settings)
    service = TranscriptionService(dataset_loader)

    service.run(stt_instance, role, split)


if __name__ == "__main__":
    app()
