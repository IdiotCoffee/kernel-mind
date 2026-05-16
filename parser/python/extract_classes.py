import ast
from pathlib import Path
from typing import List

from db.models import CodeChunk
from utils.parser_utils import extract_calls, generate_id


def extract_classes(
    tree: ast.Module,
    source: str,
    module_name: str,
    path: str,
    repo_path: str,
    imports: dict[str, str],
) -> List[CodeChunk]:
    """Extract class and method definitions from the AST tree."""
    chunks: List[CodeChunk] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_code = ast.get_source_segment(source, node)
            if not class_code:
                continue

            class_fqn = f"{module_name}.{node.name}"

            # CLASS CHUNK
            chunks.append(
                CodeChunk(
                    id=generate_id(class_fqn),
                    name=node.name,
                    type="class",
                    fqn=class_fqn,
                    module=module_name,
                    # file_path=path,
                    file_path=str(Path(path).relative_to(repo_path)),
                    parent_fqn=None,
                    code=class_code,
                    docstring=ast.get_docstring(node),
                    calls=[],  # no class-level calls
                    imports=imports,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                )
            )

            # METHODS
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_code = ast.get_source_segment(source, item)
                    if not method_code:
                        continue

                    method_fqn = f"{class_fqn}.{item.name}"
                    method_calls = extract_calls(item, class_fqn)

                    chunks.append(
                        CodeChunk(
                            id=generate_id(method_fqn),
                            name=item.name,
                            type="method",
                            fqn=method_fqn,
                            module=module_name,
                            # file_path=path,
                            file_path=str(Path(path).relative_to(repo_path)),
                            parent_fqn=class_fqn,
                            code=method_code,
                            docstring=ast.get_docstring(item),
                            calls=method_calls,
                            imports=imports,
                            # start_line=node.lineno,
                            # end_line=node.end_lineno or node.lineno,
                            start_line=item.lineno,
                            end_line=item.end_lineno or item.lineno,
                        )
                    )

    return chunks
