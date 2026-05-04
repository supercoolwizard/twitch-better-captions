from transformers import pipeline

class StudentModel:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.pipe = pipeline(
            "automatic-speech-recognition", 
            model=self.model_name, 
            device=device
        )
 
    def forward(self, input_path):
        result = self.pipe(input_path)
        return result
