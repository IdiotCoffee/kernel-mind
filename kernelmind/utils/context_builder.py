from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client.kernelmind


def build_context_pack(file_path, repo_name):
    # ---- file metadata ----
    file_doc = db.files.find_one({
        "path": file_path,
        "repo": repo_name
    })

    if not file_doc:
        return None

    # ---- code-structure elements ----
    imports = list(db.imports.find({
        "file": file_path,
        "repo": repo_name
    }))

    functions = list(db.functions.find({
        "path": file_path,
        "repo": repo_name
    }))

    classes = list(db.classes.find({
        "path": file_path,
        "repo": repo_name
    }))

    methods = list(db.methods.find({
        "path": file_path,
        "repo": repo_name
    }))

    config = db.configs.find_one({
        "file": file_path,
        "repo": repo_name
    })

    return {
        "file": file_doc,
        "imports": imports,
        "functions": functions,
        "classes": classes,
        "methods": methods,
        "config": config,      # <---- THE IMPORTANT NEW PART
    }
