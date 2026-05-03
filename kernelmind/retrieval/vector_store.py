import chromadb


class VectorStore:
    def __init__(self, path=".chromadb", collection="kernelmind_index"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_collection(collection)

    def query(self, embedding, k):
        raw = self.collection.query(
            query_embeddings=embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        return docs, metas, dists
