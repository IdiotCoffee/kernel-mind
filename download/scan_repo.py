import os

from utils.ignore_list import IGNORE_DIRS


def get_python_files(repo_path: str):
    """Recursively get all Python files in the repository, excluding ignored directories."""
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            if file.endswith(".py"):
                yield os.path.join(root, file)


def get_module_name(repo_path: str, file_path: str) -> str:
    """Get the module name for a Python file relative to the repository root."""
    rel_path = os.path.relpath(file_path, repo_path)

    # remove .py
    rel_path = rel_path.replace(".py", "")

    # remove __init__
    rel_path = rel_path.replace("__init__", "")

    # convert to dotted module
    parts = rel_path.split(os.sep)
    parts = [p for p in parts if p]

    return ".".join(parts)
