from indexing.build_repository import build_repository


def build_runtime_from_repo(
    repo_id,
    chunks,
    graph,
    device="cuda",
):
    """
    Build + persist repository runtime.

    Returns:
        RepositoryRuntime
    """

    result = build_repository(
        repo_id=repo_id,
        chunks=chunks,
        graph=graph,
        device=device,
    )

    return result["runtime"]
