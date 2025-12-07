import yaml
import hashlib
from typing import Any, Dict, List


def parse_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    file_hash = hashlib.sha256(src.encode()).hexdigest()

    try:
        data = yaml.safe_load(src)
    except yaml.YAMLError:
        return {
            "file": {
                "path": path,
                "hash": file_hash,
                "source": src,
            },
            "type": "yaml",
            "keys": [],
            "paths": [],
            "tree": None,
        }

    keys = list(data.keys()) if isinstance(data, dict) else []

    paths: List[str] = []
    extract_paths(data, prefix="", out=paths)

    return {
        "file": {
            "path": path,
            "hash": file_hash,
            "source": src,
        },
        "type": "yaml",
        "keys": keys,
        "paths": paths,
        "tree": data,
    }


def extract_paths(obj: Any, prefix: str, out: List[str]):
    """
    Flatten nested YAML structure into dot-separated paths.

    Rules:
    - Dict keys create paths
    - Lists keep the same prefix and add []
    - Scalars terminate recursion
    """

    # Dict: keys define structure
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            full = f"{prefix}.{key}" if prefix else key
            out.append(full)
            extract_paths(v, full, out)

    # List: same prefix, mark as array
    elif isinstance(obj, list):
        if prefix:
            out.append(f"{prefix}[]")
        for v in obj:
            extract_paths(v, prefix, out)

    # Scalars: stop
    else:
        return
