from kernelmind.embeddings.factory import EmbeddingFactory
from kernelmind.vector_store.chroma_store import VectorStore
import hashlib


class EmbeddingPipeline:
    def __init__(self, backend="local"):
        self.embedder = EmbeddingFactory.create(backend)
        self.store = VectorStore()

    def _chunk_id(self, repo, chunk, index):
        q = chunk.get("qualified_name") or chunk.get("name") or "file"
        return f"{repo}:{chunk['path']}:{q}:{index}"


    def _chunk_hash(self, chunk):
        return hashlib.sha256(chunk["text"].encode()).hexdigest()

    def process(self, chunks, repo_name):
        ids, docs, metas, texts = [], [], [], []

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

        embeddings = self.embedder.embed(texts)
        self.store.add(ids, embeddings, texts, metas)
