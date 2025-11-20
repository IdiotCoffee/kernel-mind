import chromadb

class VectorStore:
    def __init__(self, collection_name="kernelmind_index"):
        self.client = chromadb.PersistentClient(path=".chromadb")
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, ids, embeddings, documents, metadatas):
        # Sanitize metadata to remove None (Chroma Rust layer rejects them)
        clean_metas = []
        for meta in metadatas:
            fixed = {}
            for k, v in meta.items():
                if v is None:
                    fixed[k] = ""
                elif isinstance(v, (bool, int, float, str)):
                    fixed[k] = v
                else:
                    fixed[k] = str(v)
            clean_metas.append(fixed)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=clean_metas,
        )

    def get(self, ids):
        return self.collection.get(ids=ids)

    def query(self, text, k=5):
        return self.collection.query(
            query_texts=[text],
            n_results=k
        )
