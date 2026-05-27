import datasets
from datasets import Dataset
from datasets import load_dataset, Audio
import re
import librosa
import soundfile as sf


class DatasetTranscribeLoader:
    def __init__(self, settings, processor):
        self.settings = settings
        self.processor = processor

        self.timestamp_re = re.compile(r"\[(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)\]\s*(.*)")


    def _resolve_audio(self, example):
        return {
            **example,
            "audio": {
                "path": example["audio"]
            }
        }


    def load_dataset_for_transcription(self, split):
        dataset = load_dataset(
            "csv",
            data_files={
                split: str(
                    self.settings.DATA_DIR / split / "metadata.csv"
                )
            },
        )

        dataset = dataset.map(self._resolve_audio)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        return dataset[split]

