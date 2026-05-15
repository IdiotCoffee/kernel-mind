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

        for idx, node in enumerate(trace):
            indent = "  " * idx
            prefix = "└── " if idx > 0 else ""

            self.retrieval_log.write(f"{indent}[bold green]{prefix}{node}[/bold green]")
