from collections import defaultdict
from typing import List

from retrieval.rank import compute_query_overlap

DEBUG = False


class HybridRetriever:
    def __init__(
        self,
        embedding_retriever,
        bm25_retriever,
    ):

        self.embedding_retriever = embedding_retriever
        self.bm25_retriever = bm25_retriever

    # =====================================================
    # Reciprocal Rank Fusion
    # =====================================================

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

    # =====================================================
    # Query-Aware Seed Reranking
    # =====================================================

    def rerank_by_query_alignment(
        self,
        query: str,
        candidates: List[dict],
    ) -> List[dict]:
        """
        Re-anchor hybrid retrieval results
        to query intent BEFORE graph expansion.

        This improves:
        - workflow locality
        - semantic precision
        - seed quality
        """

        reranked = []

        for item in candidates:
            chunk = item["chunk"]

            base_score = item["score"]

            # -----------------------------------
            # Query-symbol overlap
            # -----------------------------------

            overlap_score = compute_query_overlap(
                query=query,
                fqn=chunk.fqn,
            )

            # -----------------------------------
            # Query-aware blended score
            # -----------------------------------

            final_score = base_score * 0.7 + overlap_score * 0.3

            reranked.append(
                {
                    **item,
                    "score": round(final_score, 4),
                    "overlap_score": round(overlap_score, 4),
                }
            )

        reranked.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        # -----------------------------------
        # Debug
        # -----------------------------------
        if DEBUG:
            print("\nHYBRID QUERY RERANK:\n")

            for item in reranked[:10]:
                print(
                    item["fqn"],
                    "| base =",
                    round(item["score"], 4),
                    "| overlap =",
                    item["overlap_score"],
                )

        return reranked

    # =====================================================
    # Search
    # =====================================================

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[dict]:

        # -----------------------------------
        # BM25 retrieval
        # -----------------------------------

        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=top_k,
        )

        # -----------------------------------
        # Embedding retrieval
        # -----------------------------------

        embedding_results = self.embedding_retriever.search(
            query=query,
            top_k=top_k,
        )

        # -----------------------------------
        # Hybrid fusion
        # -----------------------------------

        fused = self.reciprocal_rank_fusion(
            bm25_results=bm25_results,
            embedding_results=embedding_results,
        )

        # -----------------------------------
        # Query-aware reranking
        # -----------------------------------

        reranked = self.rerank_by_query_alignment(
            query=query,
            candidates=fused,
        )

        return reranked[:top_k]
