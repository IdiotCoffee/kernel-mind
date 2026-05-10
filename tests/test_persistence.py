import os
import shutil

# from pymongo import MongoClient
from download.scan_repo import get_python_files
from graph.build_graph import build_graph
from indexing.build_repository import build_repository
from indexing.repository_runtime import (
    RepositoryRuntime,
)
from parser.python.parser import (
    parse_python_file,
)

# =====================================================
# Config
# =====================================================

REPO_NAME = "full-stack-fastapi-template"

REPO_PATH = os.path.join(
    "repos",
    REPO_NAME,
)

# =====================================================
# Chunk Loader
# =====================================================


def load_chunks(repo_path):

    chunks = []

    files = list(get_python_files(repo_path))

    print(f"\nPython files found: {len(files)}")

    for file_path in files:
        file_chunks = parse_python_file(
            path=file_path,
            repo_path=repo_path,
        )

        chunks.extend(file_chunks)

    return chunks


# =====================================================
# Test
# =====================================================


def test_real_repository_persistence():

    # -------------------------------------------------
    # Remove old artifacts
    # -------------------------------------------------

    repo_storage_path = os.path.join(
        ".kernelmind",
        "repos",
        REPO_NAME,
    )

    if os.path.exists(repo_storage_path):
        print("\nRemoving old repository artifacts...\n")

        shutil.rmtree(repo_storage_path)

    # -------------------------------------------------
    # Mongo
    # -------------------------------------------------

    # client = MongoClient("mongodb://localhost:27017")

    # db = client["github-companion"]

    # collection = db["chunks"]

    # -------------------------------------------------
    # Parse repository
    # -------------------------------------------------

    print("\nParsing repository...\n")

    chunks = load_chunks(REPO_PATH)

    print(f"Chunks extracted: {len(chunks)}")
    # -------------------------------------------------
    # Load graph
    # -------------------------------------------------

    print("\nLoading graph...\n")

    graph = build_graph(chunks)

    print(f"Graph nodes: {len(graph)}")

    # -------------------------------------------------
    # Build repository artifacts
    # -------------------------------------------------

    print("\nBuilding repository runtime...\n")

    summary = build_repository(
        repo_id=REPO_NAME,
        chunks=chunks,
        graph=graph,
        device="cpu",
    )

    print("\nBuild summary:\n")

    print(summary)

    # -------------------------------------------------
    # Reload persisted repository
    # -------------------------------------------------

    print("\nReloading repository...\n")

    runtime = RepositoryRuntime.load(
        repo_id=REPO_NAME,
        device="cpu",
    )

    # -------------------------------------------------
    # Validate chunks
    # -------------------------------------------------

    assert len(runtime.chunks) > 0

    # -------------------------------------------------
    # Validate graph
    # -------------------------------------------------

    assert len(runtime.graph) > 0

    # -------------------------------------------------
    # Validate semantic retrieval
    # -------------------------------------------------

    results = runtime.embedding_retriever.search(
        "JWT authentication",
        top_k=3,
    )

    assert len(results) > 0

    print("\nSemantic retrieval OK\n")

    # -------------------------------------------------
    # Validate BM25 retrieval
    # -------------------------------------------------

    bm25_results = runtime.bm25_retriever.search(
        "password reset token",
        top_k=3,
    )

    assert len(bm25_results) > 0

    print("\nBM25 retrieval OK\n")

    # -------------------------------------------------
    # Final success
    # -------------------------------------------------

    print("\nREAL REPOSITORY PERSISTENCE TEST PASSED\n")


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    test_real_repository_persistence()
