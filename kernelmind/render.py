from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

console = Console()


def render_heading(text):
    console.print(f"\n[bold deep_sky_blue3]{text}[/bold deep_sky_blue3]\n")


def render_answer(answer: str):
    """Render the final synthesized answer nicely."""
    console.print(Markdown(answer))


def render_chunk(chunk):
    code = chunk.get("text", "")
    path = chunk.get("path", "<unknown>")
    start = chunk.get("start", "?")
    end = chunk.get("end", "?")

    # Detect language (only python now, can add JS/TS later)
    if path.endswith(".py"):
        language = "python"
    elif path.endswith(".js"):
        language = "javascript"
    elif path.endswith(".ts"):
        language = "typescript"
    else:
        language = "text"

    header = Text(f"{path}:{start}-{end}", style="bold blue")

    try:
        syntax = Syntax(
            code,
            language,
            line_numbers=True,
            word_wrap=False,
            theme="monokai",
        )
    except TypeError:
        # Fallback for old rich versions
        syntax = Syntax(
            code,
            language,
            line_numbers=True,
            word_wrap=False,
        )
    console.print(Panel(syntax, title=header, border_style="blue"))


def render_full_output(answer, chunks):
    render_heading("🔎 KernelMind Answer")
    render_answer(answer)

    if chunks:
        render_heading("📄 Top Retrieved Chunks")
        for c in chunks:
            render_chunk(c)
