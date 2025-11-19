
import chromadb

class VectorStore:
    def __init__(self, collection_name="kernelmind_index"):
        self.client = chromadb.PersistentClient(path=".chromadb")
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, ids, embeddings, documents, metadatas):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def get(self, ids):
        return self.collection.get(ids=ids)

    def query(self, text, k=5):
        return self.collection.query(
            query_texts=[text],
            n_results=k
        )
