from pathlib import Path

BASE_STORAGE_DIR = Path(".kernelmind/repos")


def get_repo_path(repo_id: str) -> Path:
    return BASE_STORAGE_DIR / repo_id


def ensure_repo_dirs(repo_id: str):
    repo_path = get_repo_path(repo_id)

    repo_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return repo_path
