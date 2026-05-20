from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.stt.base import BaseSTTArchitecture


class WhisperSTT(BaseSTTArchitecture):
    def __init__(self, device, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype

    def load_model(self, model_id):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id=model_id, 
            torch_dtype=self.torch_dtype,
        ).to(self.device)

        return model

    def load_processor(self, model_id):
        processor = AutoProcessor.from_pretrained(model_id)

        return processor

    def transcribe(self, model, processor, audio_path):
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        result = pipe(audio_path)

        return result["text"]


