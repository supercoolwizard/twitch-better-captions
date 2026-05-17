import pandas as pd

class TranscriptionService:
    def __init__(self, stt_instance, settings):
        self.stt_instance = stt_instance
        self.settings = settings

    def run(self):
        column = self.stt_instance.metadata_column

        for split in self.stt_instance.splits_to_process:
            metadata_path = self.settings.DATA_DIR / split / "metadata.csv"
            metadata_df = pd.read_csv(metadata_path)

            for _, row in metadata_df.iterrows():
                audio_path = row["audio_path"]
                output_path = row[column]

                text = self.stt_instance.local_hf_transcribe(audio_path)

                with open(output_path, "w") as f:
                    f.write(text)

