from textual.containers import Vertical
from textual.widgets import RichLog, Static


class LogsPanel(Vertical):
    def compose(self):

        yield Static(
            "[bold orange1]Pipeline Logs[/bold orange1]",
            markup=True,
            classes="panel-title",
        )

        self.retrieval_log = RichLog(
            highlight=True,
            markup=True,
        )

        yield self.retrieval_log

    def write_log(self, message: str):

        self.retrieval_log.write(f"[white]{message}[/white]")
