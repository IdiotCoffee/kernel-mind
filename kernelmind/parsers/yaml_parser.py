import yaml
import hashlib
from typing import Any, Dict


def parse_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    file_hash = hashlib.sha256(src.encode()).hexdigest()

    try:
        data = yaml.safe_load(src)
    except yaml.YAMLError as e:
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
