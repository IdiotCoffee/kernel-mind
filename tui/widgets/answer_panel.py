from textual.containers import Vertical
from textual.widgets import RichLog, Static


class AnswerPanel(Vertical):
    def compose(self):

        yield Static(
            "[bold orange1]Grounded Answer[/bold orange1]",
            markup=True,
            classes="panel-title",
        )

        self.retrieval_log = RichLog(
            highlight=True,
            markup=True,
        )

        yield self.retrieval_log

    def update_answer(self, answer: str):

        self.retrieval_log.clear()

        self.retrieval_log.write(f"[bold white]{answer}[/bold white]")
