import os
from kernelmind.ingestion.downloader import download_and_extract
from kernelmind.ingestion.crawler import crawl_repo

from kernelmind.parsers.python_parser import parse_python
from kernelmind.parsers.js_parser import parse_javascript   # <-- NEW

from kernelmind.utils.mongo_store import save_parsed_output
from kernelmind.utils.context_builder import build_context_pack
from kernelmind.utils.chunker import build_text_chunks
from kernelmind.embeddings.embedding_pipeline import EmbeddingPipeline
from kernelmind.search import search


def ask_repo_url():
    print("Enter GitHub repo URL:")
    return input("> ").strip()


def extract_repo_name(path):
    return os.path.basename(path)


if __name__ == "__main__":
    # ----------------------------------
    # 1. Ask for repo URL
    # ----------------------------------
    repo_url = ask_repo_url()
    print(f"\nDownloading {repo_url}...\n")

    # ----------------------------------
    # 2. Download
    # ----------------------------------
    path = download_and_extract(repo_url)
    repo_name = extract_repo_name(path)

    print(f"Downloaded to: {path}")
    print(f"Repo name    : {repo_name}\n")

    # ----------------------------------
    # 3. Crawl files
    # ----------------------------------
    files = crawl_repo(path)
    print(f"Found {len(files)} total files.\n")

    # Split by language
    py_files  = [f for f in files if f.endswith(".py")]
    js_files  = [f for f in files if f.endswith((".js", ".jsx"))]
    ts_files  = [f for f in files if f.endswith((".ts", ".tsx"))]

    print(f"{len(py_files)} Python files")
    print(f"{len(js_files)} JavaScript files")
    print(f"{len(ts_files)} TypeScript files\n")

    # ----------------------------------
    # 4. Parse & store AST
    # ----------------------------------
    print("Parsing files...\n")

    # Python
    for f in py_files:
        print(f"[PY] {f}")
        parsed = parse_python(f)
        save_parsed_output(parsed, repo_name, repo_root=path)

    # JS
    for f in js_files:
        print(f"[JS] {f}")
        parsed = parse_javascript(f)
        save_parsed_output(parsed, repo_name, repo_root=path)

    # TS
    for f in ts_files:
        print(f"[TS] {f}")
        parsed = parse_javascript(f)   # same parser handles TS
        save_parsed_output(parsed, repo_name, repo_root=path)

    # ----------------------------------
    # 5. Embedding pipeline
    # ----------------------------------
    pipeline = EmbeddingPipeline(backend="local")

    # ----------------------------------
    # 6. Chunk + embed ALL code files
    # ----------------------------------
    total_chunks = 0

    all_code_files = py_files + js_files + ts_files

    for f in all_code_files:
        logical_path = f.replace(path + "/", "")
        pack = build_context_pack(logical_path, repo_name)

        chunks = build_text_chunks(pack, repo_root=path)
        if chunks:
            print(f"Embedding {len(chunks)} chunks from {logical_path}")
            pipeline.process(chunks, repo_name)
            total_chunks += len(chunks)

    print(f"\nDONE. Embedded {total_chunks} chunks for repo '{repo_name}'.\n")

    # ----------------------------------
    # 7. Search session
    # ----------------------------------
    print("Search is ready. Type queries below (exit to quit).\n")

    while True:
        q = input("search > ").strip()
        if q.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if q:
            search(q, k=5, repo_name=repo_name)
            print("\n----------------------------------------\n")
