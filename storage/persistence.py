import json
import pickle

import faiss

from storage.paths import ensure_repo_dirs
from storage.serializer import (
    deserialize_chunk,
    deserialize_graph,
    serialize_chunk,
    serialize_graph,
)

# =========================================================
# Save Chunks
# =========================================================


def save_chunks(repo_id, chunks):

    repo_path = ensure_repo_dirs(repo_id)

    chunks_path = repo_path / "chunks.jsonl"

    with open(chunks_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(serialize_chunk(chunk)) + "\n")


def load_chunks(repo_id):

    repo_path = ensure_repo_dirs(repo_id)

    chunks_path = repo_path / "chunks.jsonl"

    chunks = []

    with open(chunks_path) as f:
        for line in f:
            chunks.append(deserialize_chunk(json.loads(line)))

    return chunks


# =========================================================
# Save Graph
# =========================================================


def save_graph(repo_id, graph):

    repo_path = ensure_repo_dirs(repo_id)

    graph_path = repo_path / "graph.json"

    with open(graph_path, "w") as f:
        json.dump(
            serialize_graph(graph),
            f,
            indent=2,
        )


def load_graph(repo_id):

    repo_path = ensure_repo_dirs(repo_id)

    graph_path = repo_path / "graph.json"

    with open(graph_path) as f:
        data = json.load(f)

    return deserialize_graph(data)


# =========================================================
# Save FAISS
# =========================================================


def save_faiss_index(
    repo_id,
    index,
    chunk_ids,
):

    repo_path = ensure_repo_dirs(repo_id)

    faiss.write_index(
        index,
        str(repo_path / "faiss.index"),
    )

    with open(
        repo_path / "chunk_ids.json",
        "w",
    ) as f:
        json.dump(chunk_ids, f)


def load_faiss_index(repo_id):

    repo_path = ensure_repo_dirs(repo_id)

    index = faiss.read_index(str(repo_path / "faiss.index"))

    with open(repo_path / "chunk_ids.json") as f:
        chunk_ids = json.load(f)

    return index, chunk_ids


# =========================================================
# Save BM25
# =========================================================


def save_bm25(repo_id, bm25):

    repo_path = ensure_repo_dirs(repo_id)

    with open(
        repo_path / "bm25.pkl",
        "wb",
    ) as f:
        pickle.dump(bm25, f)


def load_bm25(repo_id):

    repo_path = ensure_repo_dirs(repo_id)

    with open(
        repo_path / "bm25.pkl",
        "rb",
    ) as f:
        return pickle.load(f)
