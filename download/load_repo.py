import os

from git import Repo

REPO_DIR = "repos"


def clone_repo(repo_url: str) -> str:
    """Clone a repository from the given URL and return the local path."""
    if not os.path.exists(REPO_DIR):
        os.makedirs(REPO_DIR)

    repo_name = repo_url.split("/")[-1].replace(".git", "")
    path = os.path.join(REPO_DIR, repo_name)

    if os.path.exists(path):
        return path

    Repo.clone_from(repo_url, path)
    return path
