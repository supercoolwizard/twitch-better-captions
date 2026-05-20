
class WhisperArchitecture:
    def load_base_model(self, model_id):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id=model_id, 
            torch_dtype=self.torch_dtype,
        ).to(self.device)

    def prepare_inputs(self, audio):
        
