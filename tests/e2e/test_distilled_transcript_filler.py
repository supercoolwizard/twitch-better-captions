from src.stt.student_distilled import StudentModelDistilled
from src.config import settings
import pandas as pd
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv("src/local_settings.env")
hf_token = os.getenv("HF_TOKEN")
hf_username = HfApi().whoami()["name"]
hf_model_repo = f"{hf_username}/moonshine-lora-finetuned"


student_dist = StudentModelDistilled(hf_model_repo, settings.device)

metadata_path = settings.DATA_DIR / "test" / "metadata.csv"
metadata_df = pd.read_csv(metadata_path)
for index, row in metadata_df.iterrows():
    audio_path = row["audio_path"]
    student_distilled_path = row["student_distilled_path"]

    student_dist_transcript = student_dist.transcribe(audio_path)
    print(student_dist_transcript)

    with open(str(student_distilled_path), "w") as f:
        f.write(student_dist_transcript)


