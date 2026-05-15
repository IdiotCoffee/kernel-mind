from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import ProgressBar, Static


class IndexingScreen(Screen):
    def compose(self) -> ComposeResult:

        yield Static(
            "[bold orange1]Repository Indexing[/bold orange1]",
            markup=True,
        )

        yield ProgressBar(total=100)
