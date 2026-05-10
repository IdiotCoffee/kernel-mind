from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from db.models import CodeChunk


class EmbeddingRetriever:
    def __init__(
        self,
        chunks: List[CodeChunk],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
    ):
        self.chunks = chunks
        self.chunk_id_map = {idx: chunk.id for idx, chunk in enumerate(chunks)}
        # -----------------------------------
        # Load embedding model
        # -----------------------------------

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        # -----------------------------------
        # Build semantic documents
        # -----------------------------------

        self.documents = [self.build_document(chunk) for chunk in chunks]

        # -----------------------------------
        # Generate embeddings
        # -----------------------------------

        embeddings = self.model.encode(
            self.documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        self.embeddings = embeddings

        # -----------------------------------
        # Build FAISS index
        # -----------------------------------

        dimension = embeddings.shape[1]

        self.index: faiss.Index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)  # type: ignore

    def build_document(
        self,
        chunk: CodeChunk,
    ) -> str:
        """
        Build semantic representation
        for embedding generation.
        """

        parts = [
            f"FQN: {chunk.fqn}",
            f"TYPE: {chunk.type}",
        ]

        # -----------------------------------
        # Add docstring
        # -----------------------------------

        if chunk.docstring:
            parts.append(chunk.docstring)

        # -----------------------------------
        # Add code
        # -----------------------------------

        if chunk.code:
            parts.append(chunk.code)

        # -----------------------------------
        # Add call relationships
        # -----------------------------------

        if chunk.calls:
            parts.append("CALLS: " + " ".join(chunk.calls))

        return "\n".join(parts)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[dict]:

        # -----------------------------------
        # Encode query
        # -----------------------------------

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        # -----------------------------------
        # Search FAISS
        # -----------------------------------

        scores, indices = self.index.search(  # type: ignore
            query_embedding,
            top_k,
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0],
        ):
            chunk = self.chunks[idx]

            results.append(
                {
                    "score": float(score),
                    "chunk": chunk,
                }
            )

        return results
