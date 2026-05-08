from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from peft import PeftModel, PeftConfig

class StudentModelFineTuned:
    def __init__(self, adapter_model_id, device):
        self.device = device

        config = PeftConfig.from_pretrained(adapter_model_id)

        processor = AutoProcessor.from_pretrained(config.base_model_name_or_path)
        base_model = AutoModelForSpeechSeq2Seq.from_pretrained(config.base_model_name_or_path)
        # base_model.resize_token_embeddings(len(processor.tokenizer))

        base_model.resize_token_embeddings(32769)

        model = PeftModel.from_pretrained(base_model, adapter_model_id)


        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=self.device
        )

    def local_hf_transcribe(self, input_path):
        result = self.pipe(input_path)
        return result["text"]
