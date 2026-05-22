from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os 

class STTModel(ABC):
    def __init__(self, model_id, device, torch_dtype):
        load_dotenv("src/local_settings.env")
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype

    @abstractmethod
    def transcribe(self, sample):
        pass

    @abstractmethod
    def train(self, dataset, training_args, data_collator):
        pass

