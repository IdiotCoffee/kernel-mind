from textual.containers import Vertical
from textual.widgets import RichLog, Static


class RetrievalPanel(Vertical):
    def compose(self):

        yield Static(
            "[bold orange1]Retrieved Chunks[/bold orange1]",
            markup=True,
            classes="panel-title",
        )

        self.retrieval_log = RichLog(
            highlight=True,
            markup=True,
        )

        yield self.retrieval_log

    def update_results(self, results):

        self.retrieval_log.clear()

        for idx, (name, score) in enumerate(results, start=1):
            self.retrieval_log.write(
                f"[bold cyan]{idx}. {name}[/bold cyan] [white](score={score})[/white]"
            )
