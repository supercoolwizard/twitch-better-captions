from transformers import pipeline

class TeacherModel:
    def __init__(self, model_name, device):
        self.metadata_column = "teacher_path"
        self.model_name = model_name
        self.device = device
        self.pipe = pipeline(
            "automatic-speech-recognition", 
            model=self.model_name, 
            device=self.device
        )
 
    def local_hf_transcribe(self, input_path):
        result = self.pipe(input_path, return_timestamps=True)
        text = result["text"]
        return text
