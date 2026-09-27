from abc import ABC, abstractmethod


class ModelClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        ...