import time

from kernelmind.embeddings.embedding_pipeline import EmbeddingPipeline
from kernelmind.ranking.bm25 import bm25_score
from kernelmind.ranking.reranker import Reranker
from kernelmind.ranking.scorer import combine_scores
from kernelmind.retrieval.call_graph import expand_call_chain
from kernelmind.retrieval.filters import filter_candidates
from kernelmind.retrieval.vector_store import VectorStore
from kernelmind.utils.rewriter import QueryRewriter


class SearchPipeline:
    def __init__(self):
        self.embedder = EmbeddingPipeline()
        self.rewriter = QueryRewriter()
        self.store = VectorStore()
        self.reranker = Reranker()

    def run(self, query, k=5, repo_name=None):

        t0 = time.time()

        refined = self.rewriter.rewrite(query)
        print(f"[TIME] rewrite: {time.time() - t0:.2f}s")

        emb = self.embedder.embed([refined])
        t1 = time.time()
        print(f"[TIME] embed: {time.time() - t1:.2f}s")

        t2 = time.time()
        docs, metas, dists = self.store.query(emb, k * 10)
        print(f"[TIME] query: {time.time() - t2:.2f}s")

        t3 = time.time()
        candidates = filter_candidates(docs, metas, dists, repo_name, refined)
        print(f"[TIME] filter: {time.time() - t3:.2f}s")

        t4 = time.time()
        expanded = expand_call_chain(
            candidates[:k], repo_name, self.store.collection, self.embedder
        )
        print(f"[TIME] expand: {time.time() - t4:.2f}s")

        merged = expanded if expanded else candidates[:k]

        docs2 = [d for d, _, _ in merged]
        metas2 = [m for _, m, _ in merged]

        t5 = time.time()
        bm25_scores = bm25_score(refined, docs2)
        print(f"[TIME] bm25: {time.time() - t5:.2f}s")

        t6 = time.time()
        scores = combine_scores(bm25_scores, metas2)
        print(f"[TIME] combine: {time.time() - t6:.2f}s")

        t7 = time.time()
        rerank_scores = self.reranker.rerank(refined, docs2)
        print(f"[TIME] rerank: {time.time() - t7:.2f}s")

        if rerank_scores is not None:
            scores = rerank_scores

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [
            {
                "text": docs2[i],
                "path": metas2[i].get("path"),
                "start": metas2[i].get("start"),
                "end": metas2[i].get("end"),
                "type": metas2[i].get("type"),
                "qualified_name": metas2[i].get("qualified_name"),
            }
            for i in ranked
        ]
