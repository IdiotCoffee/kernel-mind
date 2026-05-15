from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static


class EvaluationScreen(Screen):
    def compose(self) -> ComposeResult:

        yield Static(
            "[bold orange1]Evaluation Dashboard[/bold orange1]\n\n"
            "Precision: 0.339\n"
            "Recall: 0.711\n\n"
            "Graph-aware retrieval active.",
            markup=True,
        )
