from ingestion.downloader import download_and_extract
from ingestion.crawler import crawl_repo
from parsers.python_parser import parse_python
from mongo_store import save_entities

if __name__ == "__main__":
    repo = "https://github.com/psf/requests"
    path = download_and_extract(repo)
    print(path)

    files = crawl_repo(path)
    print(f"found {len(files)} files")
    all_entities = []
    repo_name = path.split("/")[-1]

    for f in files:
        print(" -", f)
        if f.endswith(".py"):
            parsed = parse_python(f)

            # Convert your parser output into list of entity dicts
            # This is the tiny glue layer between parser → Mongo
            for cls in parsed["classes"]:
                all_entities.append({
                    "type": "class",
                    "file_path": f,
                    "name": cls["name"],
                    "docstring": cls.get("docstring"),
                    "methods": cls.get("methods", []),
                })

            for fn in parsed["functions"]:
                all_entities.append({
                    "type": "function",
                    "file_path": f,
                    "name": fn["name"],
                    "docstring": fn.get("docstring"),
                    "signature": fn.get("signature"),
                })

            for imp in parsed["imports"]:
                all_entities.append({
                    "type": "import",
                    "file_path": f,
                    "name": imp,
                })

    print(f"\nParsed {len(all_entities)} entities, saving to Mongo...")
    save_entities(all_entities, repo_name)
    print("Done.")
