from datetime import datetime


def build_manifest(
    repo_id: str,
    chunks,
    graph,
    embedding_model: str,
):
    return {
        "repo_id": repo_id,
        "indexed_at": datetime.utcnow().isoformat(),
        "embedding_model": embedding_model,
        "chunk_count": len(chunks),
        "graph_nodes": len(graph),
        "kernelmind_version": "2.0",
    }
