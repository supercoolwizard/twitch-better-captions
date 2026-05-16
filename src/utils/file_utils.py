import os

class FileUtils:
    @staticmethod
    def remove_ds_store(directory):
        for file_path in directory.iterdir():
            if file_path.name == ".DS_Store":
                os.remove(file_path)
