
from sentence_transformers import SentenceTransformer
from .base import EmbeddingBackend

class LocalEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model_name="BAAI/bge-base-en"):
        # sentence-transformers will download automatically
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts):
        # bge models recommend normalize_embeddings=True
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

