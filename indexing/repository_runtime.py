import json

from retrieval.bm25 import BM25Retriever
from retrieval.embeddings import EmbeddingRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.rerank import CrossEncoderReranker
from storage.manifest import build_manifest
from storage.paths import ensure_repo_dirs
from storage.persistence import (
    load_bm25,
    load_chunks,
    load_faiss_index,
    load_graph,
    save_bm25,
    save_chunks,
    save_faiss_index,
    save_graph,
)


class RepositoryRuntime:
    """
    Persistent repository retrieval runtime.

    Holds:
    - chunks
    - graph
    - FAISS
    - BM25
    - retrievers
    """

    def __init__(
        self,
        repo_id,
        chunks,
        graph,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda",
    ):

        self.repo_id = repo_id

        self.chunks = chunks
        self.chunk_lookup = {chunk.fqn: chunk for chunk in chunks}

        self.graph = graph

        self.embedding_model = embedding_model

        self.device = device

        # -----------------------------------
        # Build retrievers
        # -----------------------------------

        self.embedding_retriever = EmbeddingRetriever(
            chunks=chunks,
            model_name=embedding_model,
            device=device,
        )

        self.bm25_retriever = BM25Retriever(
            chunks=chunks,
        )

        # self.hybrid_retriever = HybridRetriever(
        #     chunks=chunks,
        #     device=device,
        # )
        self.hybrid_retriever = HybridRetriever(
            embedding_retriever=self.embedding_retriever,
            bm25_retriever=self.bm25_retriever,
        )
        self.reranker = CrossEncoderReranker(
            device=device,
        )

    # =====================================================
    # Persistence
    # =====================================================

    def save(self):

        # -----------------------------------
        # Save chunks
        # -----------------------------------

        save_chunks(
            self.repo_id,
            self.chunks,
        )

        # -----------------------------------
        # Save graph
        # -----------------------------------

        save_graph(
            self.repo_id,
            self.graph,
        )

        # -----------------------------------
        # Save FAISS
        # -----------------------------------

        save_faiss_index(
            self.repo_id,
            self.embedding_retriever.index,
            self.embedding_retriever.chunk_id_map,
        )

        # -----------------------------------
        # Save BM25
        # -----------------------------------

        save_bm25(
            self.repo_id,
            self.bm25_retriever.bm25,
        )

        # -----------------------------------
        # Save manifest
        # -----------------------------------

        manifest = build_manifest(
            repo_id=self.repo_id,
            chunks=self.chunks,
            graph=self.graph,
            embedding_model=self.embedding_model,
        )

        repo_path = ensure_repo_dirs(self.repo_id)

        with open(
            repo_path / "manifest.json",
            "w",
        ) as f:
            json.dump(
                manifest,
                f,
                indent=2,
            )

    # =====================================================
    # Load
    # =====================================================

    @classmethod
    def load(
        cls,
        repo_id,
        device="cuda",
    ):

        chunks = load_chunks(repo_id)

        graph = load_graph(repo_id)

        instance = cls(
            repo_id=repo_id,
            chunks=chunks,
            graph=graph,
            device=device,
        )

        # -----------------------------------
        # Restore FAISS
        # -----------------------------------

        (
            faiss_index,
            chunk_id_map,
        ) = load_faiss_index(repo_id)

        instance.embedding_retriever.index = faiss_index

        instance.embedding_retriever.chunk_id_map = chunk_id_map

        # -----------------------------------
        # Restore BM25
        # -----------------------------------

        instance.bm25_retriever.bm25 = load_bm25(repo_id)

        return instance
