from src.config import settings
from src.stt.student import StudentModel
from src.stt.teacher import TeacherModel
import pandas as pd


student = StudentModel(settings.student_model, settings.device)
teacher = TeacherModel(settings.teacher_model, settings.device)

for split in ["train", "test"]:
    metadata_path = settings.DATA_DIR / split / "metadata.csv"
    metadata_df = pd.read_csv(metadata_path)
    for index, row in metadata_df.iterrows():
        audio_path = row["audio_path"]
        student_path = row["student_path"]
        teacher_path = row["teacher_path"]

        student_transcript = student.local_hf_transcribe(audio_path)

        with open(str(student_path), "w") as f:
            f.write(student_transcript)

        teacher_transcript = teacher.local_hf_transcribe(audio_path)

        with open(str(teacher_path), "w") as f:
            f.write(teacher_transcript)

