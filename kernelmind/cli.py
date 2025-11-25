import os
import click

from kernelmind.ingestion.downloader import download_and_extract
from kernelmind.ingestion.crawler import crawl_repo
from kernelmind.parsers.python_parser import parse_python
from kernelmind.utils.mongo_store import save_parsed_output
from kernelmind.utils.context_builder import build_context_pack
from kernelmind.utils.chunker import build_text_chunks
from kernelmind.embeddings.embedding_pipeline import EmbeddingPipeline

# import the search function exactly as defined in your search.py
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
    py_files = [f for f in files if f.endswith(".py")]
    click.echo(f"Found {len(py_files)} Python files")

    for f in py_files:
        click.echo(f"Parsing: {f}")
        parsed = parse_python(f)
        save_parsed_output(parsed, repo_name, repo_root=path)

    pipeline = EmbeddingPipeline(backend="local")
    total_chunks = 0

    for f in py_files:
        logical_path = f.replace(path + "/", "")
        pack = build_context_pack(logical_path, repo_name)
        chunks = build_text_chunks(pack, repo_root=path)

        if chunks:
            click.echo(f"Embedding {len(chunks)} chunks from {logical_path}")
            pipeline.process(chunks, repo_name)
            total_chunks += len(chunks)

    click.echo(f"Ingestion complete. Embedded {total_chunks} chunks.")
    click.echo(f"You can now run:")
    click.echo(f"    kernelmind search \"your query\" --repo {repo_name}")


# -----------------------
# search command
# -----------------------
@cli.command()
@click.argument("query")
@click.option("--repo", default=None, help="Filter by repository name")
@click.option("-k", default=5, help="Top-k chunks to retrieve")
@click.option("--show", is_flag=True, help="Show full chunk content")
def search(query, repo, k, show):
    """
    Search indexed code.

    This calls your existing search(...) with synthesize=False so it only prints
    the retrieved chunks (search.py already handles pretty printing).
    """
    # search prints its own progress and results, so we just call it
    # synthesize=False ensures it won't call the synthesis step
    run_search(query, k=k, repo_name=repo, synthesize=False, show_chunks=show)

    # If user wants to see full chunk content but your search() doesn't respect
    # a 'show' flag, you can later add a parameter or post-process returned candidates.
    if show:
        # current search() prints code blocks by default in pretty(); if not,
        # we can modify search.py to accept show_chunks flag. For now it's a no-op.
        pass


# -----------------------
# answer command
# -----------------------
@cli.command()
@click.argument("question")
@click.option("-k", default=5, help="Number of supporting chunks")
@click.option("--repo", default=None, help="Filter by repository name")
def answer(question, k, repo):
    """
    Run the full pipeline: retrieve + synthesize.

    search(..., synthesize=True) prints the retrieved chunks and then calls
    synthesize_answer internally. It returns the final answer string.
    """
    result = run_search(question, k=k, repo_name=repo, synthesize=True)

    # your search() prints the answer already; if it returns the answer object,
    # we can optionally print or save it here.
    if result is not None:
        # If search() returns the answer string, print it cleanly.
        click.echo("")
        click.echo(result)


# register command aliases
cli.add_command(ingest, "i")
cli.add_command(search, "s")
cli.add_command(answer, "a")
if __name__ == "__main__":
    cli()
