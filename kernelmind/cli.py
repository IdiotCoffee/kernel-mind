import os
import click

from kernelmind.ingestion.downloader import download_and_extract
from kernelmind.ingestion.crawler import crawl_repo

from kernelmind.parsers.python_parser import parse_python
from kernelmind.parsers.js.js_parser import parse_javascript
from kernelmind.parsers.json_parser import parse_json
from kernelmind.parsers.yaml_parser import parse_yaml


from kernelmind.utils.mongo_store import save_parsed_code, save_parsed_config
from kernelmind.utils.context_builder import build_context_pack
from kernelmind.utils.chunker import build_text_chunks
from kernelmind.embeddings.embedding_pipeline import EmbeddingPipeline

from kernelmind.search import search as run_search
import subprocess
import shutil
from pathlib import Path
from kernelmind.config import load_config, save_config

from kernelmind.render import render_full_output
from kernelmind.search import search



# ---------------------------------------------------
# Environment setup checks
# ---------------------------------------------------

def ensure_node():
    """Check if Node.js is installed."""
    if shutil.which("node") is None:
        click.echo("❌ Node.js is required for JS/TS parsing.\nInstall from https://nodejs.org/")
        raise SystemExit(1)


def install_js_dependencies():
    """Run npm install inside kernelmind/parsers/js to install Babel deps."""
    js_dir = Path(__file__).parent / "parsers" / "js"

    pkg_json = js_dir / "package.json"
    if not pkg_json.exists():
        click.echo(f"❌ package.json missing in {js_dir}. Cannot install JS dependencies.")
        raise SystemExit(1)

    click.echo("📦 Installing JS parser dependencies (@babel/parser, @babel/traverse)...")

    try:
        subprocess.run(["npm", "install"], cwd=str(js_dir), check=True)
        click.echo("✅ JS dependencies installed.")
    except subprocess.CalledProcessError:
        click.echo("❌ Failed to run npm install. Make sure npm is installed.")
        raise SystemExit(1)


def ensure_ollama():
    """Check if Ollama is installed."""
    if shutil.which("ollama") is None:
        click.echo("❌ Ollama is required. Install from https://ollama.com")
        raise SystemExit(1)


def ensure_qwen_model(model="qwen2.5-coder:14b"):
    """Check and download the required Ollama model."""
    click.echo("🔎 Checking for required Ollama model...")

    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if model not in result.stdout:
        click.echo(f"⬇️  Pulling Ollama model: {model}")
        subprocess.run(["ollama", "pull", model], check=True)
        click.echo("✅ Model downloaded.")
    else:
        click.echo("✅ Model already installed.")


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

    pipeline = EmbeddingPipeline()
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
    """Answer a question using KernelMind's retrieval + LLM synthesis."""
    result = run_search(question, k=k, repo_name=repo, synthesize=True)

    if not result:
        click.echo("No answer generated.")
        return

    from kernelmind.render import render_full_output

    answer_text = result.get("answer")
    chunks = result.get("chunks", [])

    render_full_output(answer_text, chunks)

# -----------------------
# setup command
# -----------------------
@cli.command()
def setup():
    """Set up KernelMind (Node, JS parser deps, Ollama, Qwen model)."""

    click.echo("🔧 KernelMind setup in progress...\n")

    # Node.js
    click.echo("➡️ Checking Node.js...")
    ensure_node()

    # JS parser deps
    click.echo("➡️ Installing JS parser packages...")
    install_js_dependencies()

    # Ollama + Model
    click.echo("➡️ Checking Ollama installation...")
    ensure_ollama()

    click.echo("➡️ Checking Qwen model...")
    ensure_qwen_model()

    click.echo("\n🎉 KernelMind setup complete!")

@click.command()
@click.argument("api_key")
def set_api_key(api_key):
    """Set the OpenAI API key for cloud inference."""
    config = load_config()
    config["inference"]["api_key"] = api_key
    config["inference"]["mode"] = "cloud"
    save_config(config)
    click.echo("API key saved. Cloud mode enabled.")


@click.command()
def set_local():
    """Switch inference to local mode."""
    config = load_config()
    config["inference"]["mode"] = "local"
    save_config(config)
    click.echo("Switched to local inference.")


@cli.command()
def preview():
    """Preview KernelMind's CLI styling without calling LLMs."""
    from kernelmind.render import render_full_output

    dummy_answer = """
**Explanation:**
This is a preview of how KernelMind formats output.

**Key Points:**
- Bold text works.
- Code blocks are highlighted.
- Headings are colored.
"""

    dummy_chunks = [
        {
            "path": "src/example.py",
            "start": 10,
            "end": 25,
            "text": "def hello():\n    print('hello world')",
            "qualified_name": "hello",
            "type": "function",
        }
    ]

    render_full_output(dummy_answer, dummy_chunks)


# aliases
cli.add_command(ingest, "i")
cli.add_command(search, "s")
cli.add_command(answer, "a")
cli.add_command(setup, "setup")
cli.add_command(set_api_key)
cli.add_command(set_local)




if __name__ == "__main__":
    cli()
