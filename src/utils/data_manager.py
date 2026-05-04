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
            audio_files = [f for f in audio_dir.iterdir() if f.suffix == self.audio_file_extension]

            student_dir = self.get_path(split, "transcripts", "student")
            student_files = [f for f in student_dir.iterdir() if f.suffix == self.transcripts_files_extension]

            teacher_dir = self.get_path(split, "transcripts", "teacher")
            teacher_files = [f for f in teacher_dir.iterdir() if f.suffix == self.transcripts_files_extension]

            metadata_df = pd.DataFrame({"audio_path": audio_files, "student_path": student_files, "teacher_path": teacher_files})
            metadata_df.to_csv(current_split_dir / "metadata.csv", index=False)


    def transcripts_files_creator(self):
        """creates empty transcript fiiles, for them to be then filled"""
        for split in ["train", "test"]: 
            reference_path = self.get_path(split, "audio")
            reference_files = [f for f in reference_path.iterdir() if f.suffix == self.audio_file_extension]
            reference_names = [f.stem for f in reference_files]

            for role in ["student", "teacher"]:
                current_path = self.get_path(split, "transcripts", role)

                for name in reference_names:
                    open(current_path / f"{name}{self.transcripts_files_extension}", "x")
