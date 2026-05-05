from transformers import pipeline

class StudentModel:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
 
    def local_hf_transcribe(self, input_path):
        pipe = pipeline(
            "automatic-speech-recognition", 
            model=self.model_name, 
            device=self.device
        )

        result = pipe(input_path)
        text = result["text"]
        return text
