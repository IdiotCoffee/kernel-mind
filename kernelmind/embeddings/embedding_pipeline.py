import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only for embeddings

import hashlib

from sentence_transformers import SentenceTransformer

from kernelmind.vector_store.chroma_store import VectorStore


class EmbeddingPipeline:
    def __init__(self, model_name="BAAI/bge-base-en"):
        # Always run embeddings on CPU — they are lightweight and fast enough.
        self.model = SentenceTransformer(model_name, device="cpu")
        self.store = VectorStore()

    def _chunk_id(self, repo, chunk, index):
        q = chunk.get("qualified_name") or chunk.get("name") or "file"
        return f"{repo}:{chunk['path']}:{q}:{index}"

    def _chunk_hash(self, chunk):
        return hashlib.sha256(chunk["text"].encode()).hexdigest()

    def process(self, chunks, repo_name):
        ids, metas, texts = [], [], []

        for idx, chunk in enumerate(chunks):
            cid = self._chunk_id(repo_name, chunk, idx)
            chash = self._chunk_hash(chunk)

            ids.append(cid)
            texts.append(chunk["text"])

            meta = {
                "repo": repo_name,
                "path": chunk["path"],
                "type": chunk["type"],
                "name": chunk.get("name"),
                "qualified_name": chunk.get("qualified_name"),
                "class": chunk.get("class"),
                "start": chunk.get("start"),
                "end": chunk.get("end"),
                "hash": chash,
            }
            metas.append(meta)

        if not texts:
            return

        # Compute embeddings locally on CPU
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Store results in Chroma
        self.store.add(ids, embeddings, texts, metas)

    def embed(self, texts: list[str]):
        if not texts:
            return []

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
