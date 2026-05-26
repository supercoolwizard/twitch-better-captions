import subprocess
import os
import pandas as pd
from src.config import settings


def audio_cutter(file_path, start_time, duration):
    temp_file = "temp_output.mp3"

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", file_path,
        "-t", str(duration),
        "-c", "copy",
        temp_file
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    os.replace(temp_file, file_path)


for file in settings.INPUT_DIR.iterdir():
    if str(file)[-1] == "3":
        audio_cutter(file, "00:30:00", "00:30:00")

# for split in ["train", "test"]:
#     metadata_path = settings.DATA_DIR / split / "metadata.csv"
#     metadata_df = pd.read_csv(metadata_path)
#     for index, row in metadata_df.iterrows():
#         audio_path = row["audio_path"]
#
#         stupifier(audio_path, "00:30:00", "00:02:00")
