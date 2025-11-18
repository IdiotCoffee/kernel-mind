import ast
from typing import Any, Dict, List

def parse_python(path):
    with open(path, "r", encoding="utf-8") as f:
        src=f.read()
    try:
        tree=ast.parse(src)
    except SyntaxError:
        return {
            "path": path,
            "imports": [],
            "functions":[],
            "classes": [],
            "error":"syntax error"
        }
    imports = extract_imports(tree)
    functions = extract_functions(tree)
    classes = extract_classes(tree)
    return {
            "path": path,
            "imports": imports,
            "functions":functions,
            "classes": classes
    }
    
def extract_imports(tree: ast.AST):
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
                "args": [arg.arg for arg in node.args.args],
                "docstring": ast.get_docstring(node),
                "start_line": node.lineno,
                "end_line": node.end_lineno,
            })
    return funcs


def extract_classes(tree: ast.AST) -> List[Dict[str, Any]]:
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "docstring": ast.get_docstring(node),
                "methods": [
                    {
                        "name": n.name,
                        "args": [arg.arg for arg in n.args.args],
                        "docstring": ast.get_docstring(n),
                        "start_line": n.lineno,
                        "end_line": n.end_lineno,
                    }
                    for n in node.body if isinstance(n, ast.FunctionDef)
                ],
                "start_line": node.lineno,
                "end_line": node.end_lineno,
            })
    return classes

