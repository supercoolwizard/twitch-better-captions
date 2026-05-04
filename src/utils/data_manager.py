from pathlib import Path
import os
import numpy as np
import shutil


class DataManager:
    def __init__(self, data_path, input_path):
        self.data_path = data_path
        self.input_path = input_path

    def get_total_input_duration(self):
        pass

    def train_test_splitter(self):
        """"performs dummy split, copies smaller part to test/audio and larger part to train/audio"""
        input_files = [f for f in self.input_path.iterdir()]
        train_count = int(np.floor(len(input_files) * 0.8))

        train_files = input_files[:train_count]
        test_files = input_files[train_count:]

        train_dir = self.data_path / "train" / "audio"
        test_dir = self.data_path / "test" / "audio"

        for file in train_files:
            shutil.copy2(file, train_dir)

        for file in test_files:
            shutil.copy2(file, test_dir)
