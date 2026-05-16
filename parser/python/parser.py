import ast
from typing import List

from db.models import CodeChunk
from download.scan_repo import get_module_name
from parser.python.extract_classes import extract_classes
from parser.python.extract_functions import extract_functions
from parser.python.extract_imports import extract_imports


def parse_python_file(path: str, repo_path: str) -> List[CodeChunk]:
    """Parse a Python file and extract code chunks (functions, classes, methods, and imports)."""
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except Exception:
        return []

    imports = extract_imports(tree)
    module_name = get_module_name(repo_path, path)

    chunks: List[CodeChunk] = []

    chunks.extend(
        extract_functions(tree, source, module_name, path, repo_path, imports)
    )

    chunks.extend(extract_classes(tree, source, module_name, path, repo_path, imports))

    return chunks
