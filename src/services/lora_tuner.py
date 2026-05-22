from peft import LoraConfig, get_peft_model
from transformers import Trainer

class LoraTuner:
    def __init__(self, student):
        self.student = student
        
