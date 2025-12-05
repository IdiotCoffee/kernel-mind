import yaml
import hashlib
from typing import Any, Dict, List


def parse_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    file_hash = hashlib.sha256(src.encode()).hexdigest()

    try:
        data = yaml.safe_load(src)
    except yaml.YAMLError as e:
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

    paths = []
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
    """Flatten nested keys for YAML configs."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else k
            out.append(full)
            extract_paths(v, full, out)
    elif isinstance(obj, list):
        if prefix:
            out.append(prefix + "[]")
        for v in obj:
            extract_paths(v, prefix, out)
    else:
        if prefix:
            out.append(prefix)
