from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def load(self, path):
        
        pass

    @abstractmethod
    def preprocess(self, raw_input):
        pass

    @abstractmethod
    def predict(self, processed_input):
        pass

    @abstractmethod
    def postprocess(self, raw_output):
        pass