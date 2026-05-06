import pandas as pd

class TranscriptsManager:
    def __init__(self, data_path):
        self.data_path = data_path

    def metadata_crawler(self, split, file_name):
        """..."""
        metadata_path = self.data_path / split / "metadata.csv"
        metadata_df = pd.read_csv(metadata_path)

        result = metadata_df.loc[metadata_df['file_name'] == file_name]
        return result.to_dict(orient="records")

    def transcript_filler(self, text, file_name):
        with open(file_name, "w") as f:
            f.write(text)


