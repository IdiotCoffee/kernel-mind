from sentence_transformers import SentenceTransformer
import torch
from .base import EmbeddingBackend

class LocalEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model_name="BAAI/bge-base-en"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
