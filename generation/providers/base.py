from abc import ABC, abstractmethod
from typing import Iterator


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, stream: bool = False) -> str | Iterator[str]:
        pass
