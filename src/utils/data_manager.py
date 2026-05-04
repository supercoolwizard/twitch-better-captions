from pathlib import Path
import os
import numpy as np
import shutil
import pandas as pd


class DataManager:
    def __init__(self, file_extension, data_path, input_path):
        self.file_extension = file_extension
        self.data_path = data_path
        self.input_path = input_path
        self.train_dir = self.data_path / "train"
        self.test_dir = self.data_path / "test"
        self.train_transcripts_dir = self.train_dir / "transcripts"
        self.test_transcripts_dir = self.test_dir / "transcripts"
        self.train_audio_dir = self.train_dir / "audio"
        self.test_audio_dir = self.test_dir / "audio"

    def get_total_input_duration(self):
        pass

    def train_test_splitter(self):
        """"performs dummy split, copies smaller part to test/audio and larger part to train/audio"""
        input_files = [f for f in self.input_path.iterdir() if f.suffix == self.file_extension]
        train_count = int(np.floor(len(input_files) * 0.8))

        train_files = input_files[:train_count]
        test_files = input_files[train_count:]

        for file in train_files:
            shutil.copy2(file, self.train_audio_dir)

        for file in test_files:
            shutil.copy2(file, self.test_audio_dir)

    def metadata_maker(self):
        """creates metadata.csv that connects transcripts with audio files"""
        train_files_audio = [f for f in self.train_audio_dir.iterdir() if f.suffix == self.file_extension]
        train_files_audio = [f for f in self.train_audio_dir.iterdir() if f.suffix == self.file_extension]
        train_files_audio = [f for f in self.train_audio_dir.iterdir() if f.suffix == self.file_extension]
        metadata_df = pd.DataFrame({"audio_path": train_files_audio, "student_path": train_files_audio, "teacher_path": train_files_audio, })

        metadata_df.to_csv(self.train_audio_dir / "metadata.csv", index=False)

