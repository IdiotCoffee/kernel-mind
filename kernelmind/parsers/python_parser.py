import ast
import hashlib
from typing import Any, Dict, List


def sha256_of_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def parse_python(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    file_hash = hashlib.sha256(src.encode()).hexdigest()

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {
            "file": {"path": path, "hash": file_hash, "error": "syntax error"},
            "imports": [],
            "functions": [],
            "classes": [],
            "methods": []
        }

    imports = extract_imports(tree)
    functions = extract_functions(tree)
    classes, methods = extract_classes_and_methods(tree)

    return {
        "file": {
            "path": path,
            "hash": file_hash,
        },
        "imports": imports,
        "functions": functions,
        "classes": classes,
        "methods": methods
    }


def extract_imports(tree: ast.AST) -> List[str]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    return imports


def extract_functions(tree: ast.AST) -> List[Dict[str, Any]]:
    funcs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs.append({
                "name": node.name,
                "qualified_name": node.name,
                "args": [arg.arg for arg in node.args.args],
                "start_line": node.lineno,
                "end_line": node.end_lineno,
            })
    return funcs


def extract_classes_and_methods(tree: ast.AST):
    classes = []
    methods = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "qualified_name": node.name,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
            })

            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append({
                        "name": n.name,
                        "qualified_name": f"{node.name}.{n.name}",
                        "class": node.name,
                        "args": [arg.arg for arg in n.args.args],
                        "start_line": n.lineno,
                        "end_line": n.end_lineno,
                    })

    return classes, methods
