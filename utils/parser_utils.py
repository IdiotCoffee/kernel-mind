import ast
import hashlib


def generate_id(text: str) -> str:
    """Generate a 16-character hash ID from the given text."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def get_call_name(node):
    """Extract the name of a function call from an AST node."""
    if isinstance(node.func, ast.Name):
        return node.func.id

    elif isinstance(node.func, ast.Attribute):
        parts = []
        current = node.func

        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        if isinstance(current, ast.Name):
            parts.append(current.id)

        return list(reversed(parts))

    return None


def extract_calls(node, class_fqn: str | None = None):
    """Extract all function calls from an AST node, optionally with class context."""
    calls = []

    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = get_call_name(child)

            if not name:
                continue

            # simple call → foo()
            if isinstance(name, str):
                calls.append(name)
                continue

            # attribute chain
            if isinstance(name, list):
                # self.method(...)
                if class_fqn and name[0] == "self" and len(name) >= 2:
                    method_name = name[1]
                    calls.append(f"{class_fqn}.{method_name}")
                    continue

                # fallback
                calls.append(".".join(name))

    return list(set(calls))
