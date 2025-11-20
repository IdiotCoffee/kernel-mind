
from .local_backend import LocalEmbeddingBackend
from .cloud_backend import CloudEmbeddingBackend

class EmbeddingFactory:
    @staticmethod
    def create(backend="local"):
        if backend == "local":
            return LocalEmbeddingBackend()
        elif backend == "cloud":
            return CloudEmbeddingBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")
