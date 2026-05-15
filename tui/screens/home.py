from textual.app import ComposeResult
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, Static

from tui.screens.chat import ChatScreen


class HomeScreen(Screen):
    def compose(self) -> ComposeResult:

        with Center():
            with Vertical(id="home-container"):
                yield Static(
                    "[bold orange1]KernelMind v2[/bold orange1]",
                    markup=True,
                    id="title",
                )

                yield Static(
                    "Graph-Aware Repository Intelligence",
                    id="subtitle",
                )

                yield Static(
                    "\n• Hybrid Retrieval\n"
                    "• Graph Expansion\n"
                    "• Cross-Encoder Reranking\n"
                    "• Workflow Reconstruction\n",
                    id="features",
                )

                yield Button(
                    "Launch Repository Chat",
                    variant="primary",
                    id="launch-chat",
                )

    def on_button_pressed(self, event: Button.Pressed):

        if event.button.id == "launch-chat":
            self.app.push_screen(ChatScreen())
