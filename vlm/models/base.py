from abc import ABC, abstractmethod

class BaseVLM(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def generate(self, prompt: str, images: list) -> str:
        pass