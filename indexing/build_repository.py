from indexing.repository_runtime import RepositoryRuntime


def build_repository(
    repo_id,
    chunks,
    graph,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda",
):
    """
    Build and persist a repository index.

    Pipeline:

        chunks
        -> graph
        -> embeddings
        -> BM25
        -> persistence
    """

    # -----------------------------------
    # Build repository runtime
    # -----------------------------------

    repo_index = RepositoryRuntime(
        repo_id=repo_id,
        chunks=chunks,
        graph=graph,
        embedding_model=embedding_model,
        device=device,
    )

    # -----------------------------------
    # Persist repository artifacts
    # -----------------------------------

    repo_index.save()

    # -----------------------------------
    # Summary
    # -----------------------------------

    summary = {
        "repo_id": repo_id,
        "chunks": len(chunks),
        "graph_nodes": len(graph),
        "embedding_model": embedding_model,
    }

    return {
        "summary": summary,
        "runtime": repo_index,
    }
