from textual.containers import Vertical
from textual.widgets import RichLog, Static


class GraphPanel(Vertical):
    def compose(self):

        yield Static(
            "[bold orange1]Workflow Trace[/bold orange1]",
            markup=True,
            classes="panel-title",
        )

        self.retrieval_log = RichLog(
            highlight=True,
            markup=True,
        )

        yield self.retrieval_log

    def update_trace(self, trace):

        self.retrieval_log.clear()

        if not trace:
            self.retrieval_log.write("[red]No workflow trace available.[/red]")
            return

        for node in trace:
            name = node["fqn"].split(".")[-1]

            depth = node.get("depth", 0)

            score = node.get("score", 0)

            indent = "│   " * max(depth - 1, 0)

            prefix = "" if depth == 0 else "├── "

            line = (
                f"{indent}"
                f"[bold green]{prefix}{name}[/bold green] "
                f"[white](d={depth}, score={score})[/white]"
            )

            self.retrieval_log.write(line)
