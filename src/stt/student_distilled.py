from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from peft import PeftModel, PeftConfig
import torch
import librosa

class StudentModelDistilled:
    def __init__(self, adapter_model_id, device):
        self.metadata_column = "student_distilled_path"
        self.device = device

        config = PeftConfig.from_pretrained(adapter_model_id)

        self.processor = AutoProcessor.from_pretrained(config.base_model_name_or_path)

        base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            config.base_model_name_or_path
        )

        # during peft, adds dummy_token
        base_model.resize_token_embeddings(len(self.processor.tokenizer)+1)

        self.model = PeftModel.from_pretrained(base_model, adapter_model_id)
        self.model.to(self.device)


    def transcribe(self, input_path):
        audio, sr = librosa.load(input_path, sr=16000)

        input_features = self.processor(audio, sampling_rate=sr, return_tensors="pt").input_values
        input_features = input_features.to(self.device).to(self.model.dtype)

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
