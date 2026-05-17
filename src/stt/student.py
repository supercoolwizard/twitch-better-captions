from transformers import pipeline

class StudentModel:
    def __init__(self, model_name, device):
        self.metadata_column = "student_path"
        self.splits_to_process = ["train", "test"]
        self.model_name = model_name
        self.device = device
        self.pipe = pipeline(
            "automatic-speech-recognition", 
            model=self.model_name, 
            device=self.device
        )
 
    def transcribe(self, input_path):
        result = self.pipe(input_path)
        text = result["text"]
        return text
