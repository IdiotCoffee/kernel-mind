
import os
from ingestion.downloader import download_and_extract
from ingestion.crawler import crawl_repo
from parsers.python_parser import parse_python
from utils.mongo_store import save_parsed_output
from utils.context_builder import build_context_pack
from utils.chunker import build_text_chunks
from embeddings.embedding_pipeline import EmbeddingPipeline

if __name__ == "__main__":
    repo_url = "https://github.com/psf/requests"

    # 1. Download repo
    path = download_and_extract(repo_url)
    repo_name = os.path.basename(path)
    print("Downloaded to:", path)

    # 2. Crawl all files
    files = crawl_repo(path)
    print(f"Found {len(files)} total files.")

    # 3. Parse + store AST for all python files
    py_files = [f for f in files if f.endswith(".py")]
    print(f"{len(py_files)} Python files found.")

    for f in py_files:
        print("Parsing:", f)
        parsed = parse_python(f)
        save_parsed_output(parsed, repo_name, repo_root=path)

    # 4. Build embedding pipeline
    pipeline = EmbeddingPipeline(backend="local")

    # 5. Build + embed chunks for EACH python file
    total_chunks = 0
    for f in py_files:
        logical_path = f.replace(path + "/", "")  # map absolute → repo-relative
        pack = build_context_pack(logical_path, repo_name)

        chunks = build_text_chunks(pack, repo_root=path)
        if chunks:
            print(f"Embedding {len(chunks)} chunks from {logical_path}")
            pipeline.process(chunks, repo_name)
            total_chunks += len(chunks)

    print(f"\nDONE. Embedded total {total_chunks} chunks.")

