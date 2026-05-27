import datasets
from datasets import Dataset
from datasets import load_dataset, Audio
import re
import librosa
import soundfile as sf


class DatasetTrainLoader:
    def __init__(self, settings, processor):
        self.settings = settings
        self.processor = processor

        self.timestamp_re = re.compile(r"\[(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)\]\s*(.*)")


    def _prepare_segment(self, audio, start_s, end_s, text):
        sr = 16000

        start_idx = int(start_s * sr)
        end_idx = int(end_s * sr)

        chunk = audio[start_idx:end_idx]

        if len(chunk) == 0:
            return None

        inputs = self.processor(
            chunk,
            sampling_rate=16000,
            return_tensors="np",
        )

        labels = self.processor.tokenizer(
            text,
            truncation=True,
            max_length=448,
        ).input_ids

        return {
            "input_features": inputs.input_features[0],
            "labels": labels,
        }

    def _build_segments(self, audio_path, transcript_path):
        audio, _ = librosa.load(
            audio_path,
            sr=16000,
            mono=True,
        )

        samples = []

        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
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

                sample = self._prepare_segment(
                    audio=audio,
                    start_s=start_s,
                    end_s=end_s,
                    text=text,
                )

                if sample is not None:
                    samples.append(sample)

        return samples

    def load_dataset_for_train(self, split):
        raw = load_dataset(
            "csv",
            data_files={
                split: str(
                    self.settings.DATA_DIR / split / "metadata.csv"
                )
            },
        )[split]

        all_samples = []

        for item in raw:
            all_samples.extend(
                self._build_segments(
                    item["audio"],
                    item["teacher_path"],
                )
            )

        dataset = Dataset.from_list(all_samples)

        dataset.set_format(
            type="torch",
            columns=["input_features", "labels"],
        )

        return dataset


