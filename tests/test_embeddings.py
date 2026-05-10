import os

from download.scan_repo import get_python_files
from parser.python.parser import parse_python_file
from retrieval.embeddings import EmbeddingRetriever

REPO_NAME = "full-stack-fastapi-template"

REPO_PATH = os.path.join(
    "repos",
    REPO_NAME,
)


def load_chunks(repo_path):
    chunks = []

    for file_path in get_python_files(repo_path):
        try:
            file_chunks = parse_python_file(
                path=file_path,
                repo_path=repo_path,
            )

            chunks.extend(file_chunks)

        except Exception as e:
            print(f"Failed parsing {file_path}")
            print(e)

    return chunks


print("\nLoading chunks...")

chunks = load_chunks(REPO_PATH)

print(f"\nLoaded {len(chunks)} chunks")


print("\nBuilding retriever...")

retriever = EmbeddingRetriever(
    chunks=chunks,
    device="cuda",
)


queries = ["how is the access token created"]


for query in queries:
    print("\n")
    print("=" * 100)

    print(f"QUERY: {query}")

    results = retriever.search(
        query=query,
        top_k=5,
    )

    for i, result in enumerate(results):
        chunk = result["chunk"]

        print("\n----------------------------")

        print(f"RANK: {i + 1}")
        print(f"SCORE: {result['score']:.4f}")

        print(f"FQN: {chunk.fqn}")
        print(f"TYPE: {chunk.type}")

        print("\nCODE:\n")

        print(chunk.code[:800])

    print("\n" + "=" * 100)
