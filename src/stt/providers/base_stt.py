from abc import ABC, abstractmethod

class BaseSTTArchitecture(ABC):

    @abstractmethod
    def load_model(self, model_id):
        pass

    @abstractmethod
    def prepare_for_training(self, audio_path, transcript_path):
        pass

    @abstractmethod
    def transcribe(self, model, audio_path):
        pass

