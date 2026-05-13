import argparse
import pandas as pd
import os
from dotenv import load_dotenv
from huggingface_hub import HFApi

from src.stt.student import StudentModel
from src.stt.teacher import TeacherModel
from src.stt.student_distilled import StudentModelDistilled
from src.services.transcription_service import TranscriptionService
from src.adapters.huggingface import get_repo_name
from src.config import settings


def build_model(model_name: str):
    if model_name == "student":
        return StudentModel(settings.student_model, settings.device)

    if model_name == "teacher":
        return TeacherModel(settings.teacher_model, settings.device)

    if model_name == "student_distilled":
        return StudentModelDistilled(get_repo_name(), settings.device)


def main():
    parser = argparse.ArgumentParser(
        description="run inference on an stt model of choice (student/teacher/student_distilled)"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["student", "teacher", "student_distilled"]
    )

    args = parser.parse_args()

    stt_model = build_model(args.model)
    service = TranscriptionService(stt_model, settings)
    service.run()

if __name__ == "__main__":
    main()
