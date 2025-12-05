import os
import click

from kernelmind.ingestion.downloader import download_and_extract
from kernelmind.ingestion.crawler import crawl_repo

from kernelmind.parsers.python_parser import parse_python
from kernelmind.parsers.js_parser import parse_javascript
from kernelmind.parsers.json_parser import parse_json
from kernelmind.parsers.yaml_parser import parse_yaml


from kernelmind.utils.mongo_store import save_parsed_code, save_parsed_config
from kernelmind.utils.context_builder import build_context_pack
from kernelmind.utils.chunker import build_text_chunks
from kernelmind.embeddings.embedding_pipeline import EmbeddingPipeline

from kernelmind.search import search as run_search


def extract_repo_name(path):
    return os.path.basename(path)


@click.group()
def cli():
    """KernelMind - offline code search and synthesis."""
    pass


# -----------------------
# ingest command
# -----------------------
@cli.command()
@click.argument("repo_url")
def ingest(repo_url):
    """Download, parse, chunk, and embed a repository."""
    click.echo(f"Downloading {repo_url}...")
    path = download_and_extract(repo_url)
    repo_name = extract_repo_name(path)

    click.echo(f"Downloaded to: {path}")
    click.echo(f"Using repository name: {repo_name}")

    files = crawl_repo(path)

    # --- language detection ---
    py_files = [f for f in files if f.endswith(".py")]
    js_files = [f for f in files if f.endswith((".js", ".jsx"))]
    ts_files = [f for f in files if f.endswith((".ts", ".tsx"))]
    json_files = [f for f in files if f.endswith(".json")]
    yaml_files = [f for f in files if f.endswith((".yaml", ".yml"))]

    click.echo(f"Found {len(py_files)} Python files")
    click.echo(f"Found {len(js_files)} JS files")
    click.echo(f"Found {len(ts_files)} TS files")
    click.echo(f"Found {len(json_files)} JSON files")
    click.echo(f"Found {len(yaml_files)} YAML files")

    click.echo("\nParsing files...\n")

    # --- Parse & Store Code ---
    for f in py_files:
        click.echo(f"[PY] {f}")
        save_parsed_code(parse_python(f), repo_name, repo_root=path)

    for f in js_files:
        click.echo(f"[JS] {f}")
        save_parsed_code(parse_javascript(f), repo_name, repo_root=path)

    for f in ts_files:
        click.echo(f"[TS] {f}")
        save_parsed_code(parse_javascript(f), repo_name, repo_root=path)

    # --- Parse & Store Config ---
    for f in json_files:
        click.echo(f"[JSON] {f}")
        save_parsed_config(parse_json(f), repo_name, repo_root=path)

    for f in yaml_files:
        click.echo(f"[YAML] {f}")
        save_parsed_config(parse_yaml(f), repo_name, repo_root=path)

    # --- Embedding ---
    from kernelmind.utils.config_chunker import build_config_chunks
    from kernelmind.utils.mongo_store import db

    pipeline = EmbeddingPipeline(backend="local")
    total_chunks = 0

    # ---------- CODE CHUNKING ----------
    code_files = py_files + js_files + ts_files

    for f in code_files:
        logical = f.replace(path + "/", "")
        pack = build_context_pack(logical, repo_name)

        if not pack:
            continue

        chunks = build_text_chunks(pack, repo_root=path)

        if chunks:
            click.echo(f"Embedding {len(chunks)} code chunks from {logical}")
            pipeline.process(chunks, repo_name)
            total_chunks += len(chunks)

    # ---------- CONFIG CHUNKING ----------
    config_files = json_files + yaml_files

    for f in config_files:
        logical = f.replace(path + "/", "")

        # IMPORTANT: configs live in db.configs, not db.files
        config_doc = db.configs.find_one({
            "file": logical,
            "repo": repo_name
        })

        if not config_doc:
            continue

        chunks = build_config_chunks(config_doc, repo=repo_name)

        if chunks:
            click.echo(f"Embedding {len(chunks)} config chunks from {logical}")
            pipeline.process(chunks, repo_name)
            total_chunks += len(chunks)

    click.echo(f"\nIngestion complete. Embedded {total_chunks} chunks.")
    click.echo(f"You can now run: km s \"your query\" --repo {repo_name}")


# -----------------------
# search command
# -----------------------
@cli.command()
@click.argument("query")
@click.option("--repo", default=None, help="Filter by repository name")
@click.option("-k", default=5, help="Top-k chunks to retrieve")
@click.option("--show", is_flag=True, help="Show full chunk content")
def search(query, repo, k, show):

    run_search(query, k=k, repo_name=repo, synthesize=False, show_chunks=show)

    if show:
        pass


# -----------------------
# answer command
# -----------------------
@cli.command()
@click.argument("question")
@click.option("-k", default=5, help="Number of supporting chunks")
@click.option("--repo", default=None, help="Filter by repository name")
def answer(question, k, repo):

    result = run_search(question, k=k, repo_name=repo, synthesize=True)

    if result is not None:
        click.echo("")
        click.echo(result)


# aliases
cli.add_command(ingest, "i")
cli.add_command(search, "s")
cli.add_command(answer, "a")

if __name__ == "__main__":
    cli()
