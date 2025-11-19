
from embeddings.factory import EmbeddingFactory
from vector_store.chroma_store import VectorStore
import hashlib

class EmbeddingPipeline:
    def __init__(self, backend="local"):
        self.embedder = EmbeddingFactory.create(backend)
        self.store = VectorStore()

    def _chunk_id(self, repo, chunk, index):
        name = chunk.get("name", "file")
        return f"{repo}:{chunk['path']}:{chunk['type']}:{name}:{index}"

    def _chunk_hash(self, chunk):
        return hashlib.sha256(chunk["text"].encode()).hexdigest()

    def process(self, chunks, repo_name):
        ids = []
        docs = []
        metas = []
        texts = []

        for idx,chunk in enumerate(chunks):
            cid = self._chunk_id(repo_name, chunk, idx)
            chash = self._chunk_hash(chunk)

            ids.append(cid)
            texts.append(chunk["text"])

            raw_meta = {
                "repo": repo_name,
                "path": chunk["path"],
                "type": chunk["type"],
                "name": chunk.get("name"),
                "class": chunk.get("class"),
                "hash": chash,
            }

            # Remove keys with None values
            clean_meta = {k: v for k, v in raw_meta.items() if v is not None}
            metas.append(clean_meta)

        if not texts:
            return

        embeddings = self.embedder.embed(texts)
        self.store.add(ids, embeddings, texts, metas)

