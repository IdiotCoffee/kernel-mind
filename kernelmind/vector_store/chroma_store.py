import chromadb

class VectorStore:
    def __init__(self, collection_name="kernelmind_index"):
        self.client = chromadb.PersistentClient(path=".chromadb")
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, ids, embeddings, documents, metadatas):
        """
        Adds vectors in safe batches. Chroma cannot handle > ~5461 items per batch.
        """

        # Clean metadata to satisfy Chroma restrictions
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

        # Chroma batch safety margin
        BATCH = 2000

        # Add in slices
        for i in range(0, len(ids), BATCH):
            j = i + BATCH
            self.collection.add(
                ids=ids[i:j],
                embeddings=embeddings[i:j],
                documents=documents[i:j],
                metadatas=clean_metas[i:j],
            )

    def get(self, ids):
        return self.collection.get(ids=ids)

    def query(self, text, k=5):
        return self.collection.query(
            query_texts=[text],
            n_results=k
        )
    