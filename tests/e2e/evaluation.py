from src.stt.student_fine_tuned import StudentModelFineTuned
from src.config import settings
from src.utils.transcripts_manager import TranscriptsManager
import pandas as pd

tm = TranscriptsManager(settings.DATA_DIR)


student_ft = StudentModelFineTuned(str(settings.MODELS_DIR), settings.device)

metadata_path = settings.DATA_DIR / "test" / "metadata.csv"
metadata_df = pd.read_csv(metadata_path)
for index, row in metadata_df.iterrows():
    audio_path = row["audio_path"]

    student_ft_transcript = student_ft.transcribe(audio_path)
    print(student_ft_transcript)
