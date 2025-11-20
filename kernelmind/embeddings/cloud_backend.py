
from .base import EmbeddingBackend

class CloudEmbeddingBackend(EmbeddingBackend):
    def embed(self, texts):
        raise RuntimeError("Cloud embeddings not configured yet.")
