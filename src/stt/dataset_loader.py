import datasets
from datasets import Dataset
from datasets import load_dataset, Audio
import re
import soundfile as sf


class DatasetLoader:
    def __init__(self, settings):
        self.settings = settings
        self.timestamp_re = re.compile(r"\[(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)\]\s*(.*)")


    def _resolve_audio(self, example):
        return {
            **example,
            "audio": {
                "path": example["audio"]
            }
        }

    def _build_segments(self, audio_path, transcript_path):
        audio, sr = sf.read(audio_path)

        samples = []

        with open(transcript_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            match = self.timestamp_re.match(line)

            if not match:
                continue

            start_s = float(match.group(1))
            end_s = float(match.group(2))
            text = match.group(3).strip()

            if not text:
                continue

            start_idx = int(start_s * sr)
            end_idx = int(end_s * sr)

            chunk = audio[start_idx:end_idx]

            samples.append({
                "audio": {
                    "array": chunk,
                    "sampling_rate": sr
                },
                "text": text
            })

        return samples


    def load_dataset_for_transcription(self, split):
        dataset = load_dataset(
            "csv", 
            data_files={
                split: str(self.settings.DATA_DIR / split / "metadata.csv")
            }
        )
        dataset = dataset.map(self._resolve_audio)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))


        return dataset[split]


    def load_dataset_for_train(self, split):
        raw = load_dataset(
            "csv", 
            data_files={
                split: str(self.settings.DATA_DIR / split / "metadata.csv")
            }
        )[split]

        all_samples = []

        for item in raw:
            all_samples.extend(
                self._build_segments(
                    item["audio"],
                    item["teacher_path"]
                )
            )

        dataset = Dataset.from_list(all_samples)

        return dataset
 


