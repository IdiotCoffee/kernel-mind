import os
from datetime import datetime
from pymongo import MongoClient
from copy import deepcopy

client = MongoClient("mongodb://localhost:27017")
db = client.kernelmind
def normalize_keys(obj):
    """Recursively ensure all dict keys are strings."""
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            key = str(k)
            new[key] = normalize_keys(v)
        return new
    elif isinstance(obj, list):
        return [normalize_keys(i) for i in obj]
    else:
        return obj


# ============================================================
# 1. CODE FILE STORAGE (Python, JS, TS)
# ============================================================

def save_parsed_code(parsed, repo_name, repo_root=None):
    """
    Stores:
      - db.files
      - db.imports
      - db.functions
      - db.classes
      - db.methods
    """

    file_doc = parsed["file"].copy()

    # Attach source if parser provided it
    if "source" in parsed:
        file_doc["source"] = parsed["source"]

    # Normalize file path
    if repo_root:
        file_doc["path"] = os.path.relpath(file_doc["path"], repo_root)

    file_doc.update({
        "repo": repo_name,
        "created_at": datetime.utcnow(),
        "type": "code"
    })

    # Check if changed
    existing = db.files.find_one({"path": file_doc["path"], "repo": repo_name})
    file_changed = (not existing) or (existing["hash"] != file_doc["hash"])

    # Upsert metadata only if changed
    if file_changed:
        db.files.update_one(
            {"path": file_doc["path"], "repo": repo_name},
            {"$set": file_doc},
            upsert=True,
        )

    # Clear old metadata
    db.imports.delete_many({"file": file_doc["path"], "repo": repo_name})
    db.functions.delete_many({"file": file_doc["path"], "repo": repo_name})
    db.classes.delete_many({"file": file_doc["path"], "repo": repo_name})
    db.methods.delete_many({"file": file_doc["path"], "repo": repo_name})

    now = datetime.utcnow()

    # Insert imports
    for imp in parsed.get("imports", []):
        db.imports.insert_one({
            "file": file_doc["path"],
            "repo": repo_name,
            "import": imp,
            "hash": file_doc["hash"],
            "created_at": now,
        })

    # Insert functions
    for fn in parsed.get("functions", []):
        fn_doc = deepcopy(fn)
        fn_doc.update({
            "path": file_doc["path"],
            "repo": repo_name,
            "hash": file_doc["hash"],
            "created_at": now,
        })
        db.functions.insert_one(fn_doc)

    # Insert classes
    for cls in parsed.get("classes", []):
        cls_doc = deepcopy(cls)
        cls_doc.update({
            "path": file_doc["path"],
            "repo": repo_name,
            "hash": file_doc["hash"],
            "created_at": now,
        })
        db.classes.insert_one(cls_doc)

    # Insert methods
    for m in parsed.get("methods", []):
        m_doc = deepcopy(m)
        m_doc.update({
            "path": file_doc["path"],
            "repo": repo_name,
            "hash": file_doc["hash"],
            "created_at": now,
        })
        db.methods.insert_one(m_doc)

    return file_doc["path"]



# ============================================================
# 2. CONFIG FILE STORAGE (JSON, YAML, TOML)
# ============================================================

def save_parsed_config(parsed, repo_name, repo_root=None):
    """
    Stores a config entry in db.configs.
    """

    file_doc = parsed["file"].copy()

    # normalize path
    if repo_root:
        file_doc["path"] = os.path.relpath(file_doc["path"], repo_root)

    file_doc.update({
        "repo": repo_name,
        "created_at": datetime.utcnow(),
        "type": "config",
        "source": parsed["file"].get("source"),
    })

    # Clean existing record
    db.configs.delete_many({
        "file": file_doc["path"],
        "repo": repo_name
    })

    # ---- FIX: Normalize everything ----
    norm_keys = [str(k) for k in parsed.get("keys", [])]
    norm_paths = [str(p) for p in parsed.get("paths", [])]
    norm_tree = normalize_keys(parsed.get("tree"))

    # Insert
    db.configs.insert_one({
        "file": file_doc["path"],
        "repo": repo_name,
        "hash": file_doc["hash"],
        "created_at": datetime.utcnow(),
        "type": parsed.get("type"),
        "keys": norm_keys,
        "paths": norm_paths,
        "tree": norm_tree,                 # <---- normalized!
        "source": parsed["file"].get("source"),
    })

    return file_doc["path"]
