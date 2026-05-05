from transformers import pipeline

class TeacherModel:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
 
    def local_hf_transcribe(self, input_path):
        pipe = pipeline(
            "automatic-speech-recognition", 
            model=self.model_name, 
            device=self.device
        )

        result = pipe(input_path, return_timestamps=True)
        text = result["text"]
        return text
