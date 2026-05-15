from pathlib import Path
import os
import numpy as np
import shutil
import pandas as pd


class DataManager:
    def __init__(self, audio_file_extension, transcripts_file_extension, data_path, input_path):
        self.audio_file_extension = audio_file_extension
        self.transcripts_files_extension = transcripts_file_extension
        self.data_path = data_path
        self.input_path = input_path
        self.dirs = {
            "train": self.data_path / "train",
            "test": self.data_path / "test"
        }


    def get_path(self, split, category, role=None):
        """e.g. self.get_path('train','transcripts','teacher')"""
        path = self.dirs[split] / category
        if role:
            path = path / role

        path.mkdir(parents=True, exist_ok=True)
        return path


    def get_total_input_duration(self):
        pass


    def train_test_splitter(self):
        """"performs dummy split, copies smaller part to test/audio and larger part to train/audio"""
        input_files = [f for f in self.input_path.iterdir() if f.suffix == self.audio_file_extension]
        train_count = int(np.floor(len(input_files) * 0.8))

        files_split = {
            "train": input_files[:train_count],
            "test": input_files[train_count:]
        }

        for split, files in files_split.items():
            dest = self.get_path(split, "audio")
            for file in files:
                shutil.copy2(file, dest)


    def metadata_maker(self):
        """creates metadata.csv that connects transcripts with audio files"""
        for split in ["train", "test"]:
            current_split_dir = self.dirs[split]

            audio_dir = self.get_path(split, "audio")
            student_dir = self.get_path(split, "transcripts", "student")
            teacher_dir = self.get_path(split, "transcripts", "teacher")

            data = []

            for audio_path in audio_dir.iterdir():
                if audio_path.suffix != self.audio_file_extension:
                    continue

                file_name = audio_path.stem

                s_trans = student_dir / f"{file_name}{self.transcripts_files_extension}"
                t_trans = teacher_dir / f"{file_name}{self.transcripts_files_extension}"

                row = {
                    "file_name": file_name,
                    "audio_path": audio_path,
                    "student_path": s_trans,
                    "teacher_path": t_trans
                }

                if split == "test":
                    distilled_dir = self.get_path(split, "transcripts", "student_distilled")
                    d_trans = distilled_dir / f"{file_name}{self.transcripts_files_extension}"
                    row["student_distilled_path"] = d_trans

                if s_trans.exists() and t_trans.exists():
                    data.append(row)
                else:
                    print(f"Error, no transcript for {file_name}")

            metadata_df = pd.DataFrame(data)
            metadata_df.to_csv(current_split_dir / "metadata.csv", index=False)


    def transcripts_files_creator(self):
        """creates empty transcript files, for them to be then filled"""
        for split in ["train", "test"]: 
            reference_path = self.get_path(split, "audio")
            reference_files = [f for f in reference_path.iterdir() if f.suffix == self.audio_file_extension]
            reference_names = [f.stem for f in reference_files]

            roles = ["student", "teacher"]
            if split == "test":
                roles.append("student_distilled")

            for role in roles:
                current_dir = self.get_path(split, "transcripts", role)

                for name in reference_names:
                    current_path = current_dir / f"{name}{self.transcripts_files_extension}"
                    with open(current_path, 'w'):
                        pass
