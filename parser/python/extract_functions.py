import ast
from pathlib import Path
from typing import List

from db.models import CodeChunk
from utils.parser_utils import extract_calls, generate_id


def extract_functions(
    tree: ast.Module,
    source: str,
    module_name: str,
    path: str,
    repo_path: str,
    imports: dict[str, str],
) -> List[CodeChunk]:
    """Extract function definitions from the AST tree."""
    chunks: List[CodeChunk] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            code = ast.get_source_segment(source, node)
            if not code:
                continue

            fqn = f"{module_name}.{node.name}"
            calls = extract_calls(node)

            chunks.append(
                CodeChunk(
                    id=generate_id(fqn),
                    name=node.name,
                    type="function",
                    fqn=fqn,
                    module=module_name,
                    # file_path=path,
                    file_path=str(Path(path).relative_to(repo_path)),
                    parent_fqn=None,
                    code=code,
                    docstring=ast.get_docstring(node),
                    calls=calls,
                    imports=imports,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                )
            )

    return chunks
