import ast
import hashlib
from typing import Any, Dict, List

def parse_python(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    file_hash = hashlib.sha256(src.encode()).hexdigest()
    lines = src.splitlines()

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {
            "file": {"path": path, "hash": file_hash, "error": "syntax error"},
            "imports": [],
            "functions": [],
            "classes": [],
            "methods": [],
            "source": src,
        }

    imports = extract_imports(tree)
    functions = extract_functions(tree, lines)
    classes, methods = extract_classes_and_methods(tree, lines)

    return {
        "file": {"path": path, "hash": file_hash},
        "imports": imports,
        "functions": functions,
        "classes": classes,
        "methods": methods,
        "source": src,   # <-- ⭐ CRITICAL ADDITION
    }


def extract_imports(tree: ast.AST) -> List[str]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.extend(f"{module}.{a.name}" for a in node.names)
    return imports


def slice_code(lines, start, end):
    return "\n".join(lines[start - 1 : end])


def extract_functions(tree: ast.AST, lines: List[str]) -> List[Dict[str, Any]]:
    funcs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs.append({
                "name": node.name,
                "qualified_name": node.name,
                "args": [a.arg for a in node.args.args],
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "code": slice_code(lines, node.lineno, node.end_lineno),
            })
    return funcs


def extract_classes_and_methods(tree: ast.AST, lines: List[str]):
    classes = []
    methods = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "qualified_name": node.name,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "code": slice_code(lines, node.lineno, node.end_lineno),
            })

            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append({
                        "name": n.name,
                        "qualified_name": f"{node.name}.{n.name}",
                        "class": node.name,
                        "args": [a.arg for a in n.args.args],
                        "start_line": n.lineno,
                        "end_line": n.end_lineno,
                        "code": slice_code(lines, n.lineno, n.end_lineno),
                    })

    return classes, methods
