
from sentence_transformers import SentenceTransformer
from .base import EmbeddingBackend

class LocalEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)
