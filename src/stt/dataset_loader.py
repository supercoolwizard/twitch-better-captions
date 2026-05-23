import datasets
from datasets import load_dataset, Audio


class DatasetLoader:
    def __init__(self, settings):
        self.settings = settings


    def _resolve_audio(self, example):
        return {
            **example,
            "audio": {
                "path": example["audio"]
            }
        }


    def load_dataset(self, split):
        dataset = load_dataset(
            "csv", 
            data_files={
                split: str(self.settings.DATA_DIR / split / "metadata.csv")
            }
        )
        dataset = dataset.map(self._resolve_audio)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))


        return dataset[split]


