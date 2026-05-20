
from abc import ABC, abstractmethod


class BaseSTTArchitecture(ABC):
    @abstractmethod
    def load_model(self, model_id, device, torch_dtype):
        pass

    
