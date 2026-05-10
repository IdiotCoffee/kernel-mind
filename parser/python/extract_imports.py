import ast
from typing import Dict


def extract_imports(tree) -> Dict[str, str]:
    """
    Extracts import statements from the AST tree and returns a mapping of alias to full import path.

    Returns:
    {
        "urlparse": "urllib.parse.urlparse",
        "np": "numpy",
        "requests": "requests"
    }
    """

    import_map = {}

    for node in ast.walk(tree):
        # import numpy as np
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                asname = alias.asname or name.split(".")[-1]
                import_map[asname] = name

        # from urllib.parse import urlparse
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    name = alias.name
                    asname = alias.asname or name
                    import_map[asname] = f"{node.module}.{name}"

    return import_map
