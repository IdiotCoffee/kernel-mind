import os

# from pymongo import MongoClient
from download.scan_repo import get_python_files
from parser.python.parser import parse_python_file
from retrieval.bm25 import BM25Retriever

REPO_NAME = "full-stack-fastapi-template"

REPO_PATH = os.path.join(
    "repos",
    REPO_NAME,
)


def load_chunks(repo_path):
    chunks = []

    files = list(get_python_files(repo_path))

    print(f"\nPython files found: {len(files)}")

    for file_path in files:
        print(f"\nParsing: {file_path}")

        file_chunks = parse_python_file(
            path=file_path,
            repo_path=repo_path,
        )

        print(f"Extracted chunks: {len(file_chunks)}")

        chunks.extend(file_chunks)

    return chunks


print("\nLoading chunks...")

chunks = load_chunks(REPO_PATH)

print(f"\nLoaded chunks: {len(chunks)}")


retriever = BM25Retriever(
    chunks=chunks,
)


query = "password reset token"


results = retriever.search(
    query=query,
    top_k=15,
)


print("\nBM25 RESULTS\n")


for item in results:
    chunk = item["chunk"]

    print("=" * 60)

    print("FQN:", chunk.fqn)

    print("TYPE:", chunk.type)

    print("BM25 SCORE:", item["score"])

    print("\nCODE:\n")

    print(chunk.code[:600])
