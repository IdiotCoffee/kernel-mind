from kernelmind.pipeline.search_pipeline import SearchPipeline
from kernelmind.response_engine.engine import ResponseEngine

_pipeline = SearchPipeline()
_engine = ResponseEngine()


def search(query, k=5, repo_name=None):
    chunks = _pipeline.run(query, k=k, repo_name=repo_name)

    if not chunks:
        return {"answer": "No results found.", "chunks": []}

    answer = _engine.generate(query, chunks)

    return {
        "answer": answer,
        "chunks": chunks,
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Query: ")
    print(search(q))
