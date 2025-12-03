import json
import subprocess
import hashlib
from typing import Any, Dict, List


def sha256_of_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def parse_javascript(path: str) -> Dict[str, Any]:
    """Parse JS/TS via Babel parser (Node script)."""

    try:
        result = subprocess.check_output(
            ["node", "parse_js.js", path],
            stderr=subprocess.STDOUT
        ).decode("utf8")
    except subprocess.CalledProcessError as e:
        return {
            "file": {"path": path, "hash": sha256_of_file(path)},
            "error": f"Failed to parse: {e.output.decode('utf8')}",
            "imports": [],
            "functions": [],
            "classes": [],
            "methods": []
        }

    data = json.loads(result)

    if not data.get("ok"):
        return {
            "file": {"path": path, "hash": sha256_of_file(path)},
            "error": data.get("error", "Unknown error"),
            "imports": [],
            "functions": [],
            "classes": [],
            "methods": []
        }

    ast = data["ast"]

    imports = extract_imports(ast)
    functions = extract_functions(ast)
    classes, methods = extract_classes_and_methods(ast)

    return {
        "file": {"path": path, "hash": sha256_of_file(path)},
        "imports": imports,
        "functions": functions,
        "classes": classes,
        "methods": methods
    }


# ----------------------------------------------------------------------
# Extract imports
# ----------------------------------------------------------------------

def extract_imports(ast) -> List[str]:
    modules = []

    def visit(node):
        if isinstance(node, dict):
            # ES6 import
            if node.get("type") == "ImportDeclaration":
                if node.get("source", {}).get("value"):
                    modules.append(node["source"]["value"])

            # require("x")
            if node.get("type") == "CallExpression":
                callee = node.get("callee", {})
                if callee.get("type") == "Identifier" and callee.get("name") == "require":
                    args = node.get("arguments", [])
                    if args and args[0].get("type") == "StringLiteral":
                        modules.append(args[0]["value"])

            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    visit(v)

        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(ast)
    return modules


# ----------------------------------------------------------------------
# Extract functions
# ----------------------------------------------------------------------

def extract_functions(ast):
    funcs = []

    def visit(node):
        if isinstance(node, dict):
            t = node.get("type")

            # function foo() {}
            if t == "FunctionDeclaration":
                name = (node.get("id") or {}).get("name", "<anonymous>")
                funcs.append({
                    "name": name,
                    "qualified_name": name,
                    "args": [p["name"] for p in node.get("params", []) if p.get("name")],
                    "start_line": node["loc"]["start"]["line"],
                    "end_line": node["loc"]["end"]["line"],
                })

            # const foo = () => {}
            if t == "VariableDeclarator":
                id_node = node.get("id")
                init = node.get("init")

                if id_node and init and init.get("type") in ("ArrowFunctionExpression", "FunctionExpression"):
                    name = id_node.get("name", "<anonymous>")
                    params = init.get("params", [])
                    funcs.append({
                        "name": name,
                        "qualified_name": name,
                        "args": [p["name"] for p in params if p.get("name")],
                        "start_line": init["loc"]["start"]["line"],
                        "end_line": init["loc"]["end"]["line"],
                    })

            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    visit(v)

        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(ast)
    return funcs


# ----------------------------------------------------------------------
# Extract classes + methods
# ----------------------------------------------------------------------

def extract_classes_and_methods(ast):
    classes = []
    methods = []

    def visit(node, current_class=None):
        if isinstance(node, dict):
            t = node.get("type")

            # class Foo {}
            if t == "ClassDeclaration":
                cls_name = (node.get("id") or {}).get("name", "<anonymous>")
                classes.append({
                    "name": cls_name,
                    "qualified_name": cls_name,
                    "start_line": node["loc"]["start"]["line"],
                    "end_line": node["loc"]["end"]["line"],
                })
                current_class = cls_name

            # Foo { method() {} }
            if t == "ClassMethod" and current_class:
                mname = (node.get("key") or {}).get("name", "<anonymous>")
                methods.append({
                    "name": mname,
                    "qualified_name": f"{current_class}.{mname}",
                    "class": current_class,
                    "args": [p["name"] for p in node.get("params", []) if p.get("name")],
                    "start_line": node["loc"]["start"]["line"],
                    "end_line": node["loc"]["end"]["line"],
                })

            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    visit(v, current_class)

        elif isinstance(node, list):
            for item in node:
                visit(item, current_class)

    visit(ast)
    return classes, methods
