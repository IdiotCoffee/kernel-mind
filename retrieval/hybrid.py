from collections import defaultdict
from typing import List

from db.models import CodeChunk
from retrieval.bm25 import BM25Retriever
from retrieval.embeddings import EmbeddingRetriever


class HybridRetriever:
    def __init__(
        self,
        chunks: List[CodeChunk],
        device: str = "cuda",
    ):

        self.chunks = chunks

        self.embedding_retriever = EmbeddingRetriever(
            chunks=chunks,
            device=device,
        )

        self.bm25_retriever = BM25Retriever(
            chunks=chunks,
        )

    def reciprocal_rank_fusion(
        self,
        bm25_results: List[dict],
        embedding_results: List[dict],
        k: int = 60,
    ) -> List[dict]:
        """
        Reciprocal Rank Fusion (RRF)

        Combines rankings from:
        - BM25
        - embeddings
        """

        scores = defaultdict(float)

        chunk_lookup = {}

        # -----------------------------------
        # BM25 contribution
        # -----------------------------------

        for rank, item in enumerate(bm25_results):
            chunk = item["chunk"]

            chunk_lookup[chunk.fqn] = chunk

            scores[chunk.fqn] += 1 / (k + rank + 1)

        # -----------------------------------
        # Embedding contribution
        # -----------------------------------

        for rank, item in enumerate(embedding_results):
            chunk = item["chunk"]

            chunk_lookup[chunk.fqn] = chunk

            scores[chunk.fqn] += 1 / (k + rank + 1)

        fused = []

        for fqn, score in scores.items():
            fused.append(
                {
                    "fqn": fqn,
                    "score": score,
                    "chunk": chunk_lookup[fqn],
                }
            )

        fused.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return fused

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[dict]:

        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=top_k,
        )

        embedding_results = self.embedding_retriever.search(
            query=query,
            top_k=top_k,
        )

        fused = self.reciprocal_rank_fusion(
            bm25_results=bm25_results,
            embedding_results=embedding_results,
        )

        return fused[:top_k]
