import os
from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017")
db = client.kernelmind


def save_parsed_output(parsed, repo_name, repo_root=None):
    file_doc = parsed["file"]

    # If repo_root provided, enforce relative path
    if repo_root:
        file_doc["path"] = os.path.relpath(file_doc["path"], repo_root)

    file_doc["repo"] = repo_name
    file_doc["created_at"] = datetime.utcnow()

    existing = db.files.find_one({
        "path": file_doc["path"],
        "repo": repo_name
    })

    if existing and existing["hash"] == file_doc["hash"]:
        return

    db.files.update_one(
        {"path": file_doc["path"], "repo": repo_name},
        {"$set": file_doc},
        upsert=True
    )

    db.imports.delete_many({"file": file_doc["path"], "repo": repo_name})
    db.functions.delete_many({"file": file_doc["path"], "repo": repo_name})
    db.classes.delete_many({"file": file_doc["path"], "repo": repo_name})
    db.methods.delete_many({"file": file_doc["path"], "repo": repo_name})

    for imp in parsed["imports"]:
        db.imports.insert_one({
            "file": file_doc["path"],
            "repo": repo_name,
            "import": imp,
            "hash": file_doc["hash"],
            "created_at": datetime.utcnow()
        })

    for fn in parsed["functions"]:
        fn["path"] = file_doc["path"]
        fn["repo"] = repo_name
        fn["hash"] = file_doc["hash"]
        fn["created_at"] = datetime.utcnow()
        db.functions.insert_one(fn)

    for cls in parsed["classes"]:
        cls["path"] = file_doc["path"]
        cls["repo"] = repo_name
        cls["hash"] = file_doc["hash"]
        cls["created_at"] = datetime.utcnow()
        db.classes.insert_one(cls)

    for m in parsed["methods"]:
        m["path"] = file_doc["path"]
        m["repo"] = repo_name
        m["hash"] = file_doc["hash"]
        m["created_at"] = datetime.utcnow()
        db.methods.insert_one(m)
