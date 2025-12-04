import json
import hashlib
from typing import Any, Dict


def parse_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    file_hash = hashlib.sha256(src.encode()).hexdigest()

    try:
        data = json.loads(src)
    except json.JSONDecodeError as e:
        return {
            "file": {"path": path, "hash": file_hash, "error": str(e)},
            "imports": [],
            "functions": [],
            "classes": [],
            "methods": []
        }

    # Extract top-level keys
    keys = list(data.keys()) if isinstance(data, dict) else []

    return {
        "file": {
            "path": path,
            "hash": file_hash,
            "keys": keys,
        },
        "imports": [],
        "functions": [],
        "classes": [],
        "methods": []
    }
