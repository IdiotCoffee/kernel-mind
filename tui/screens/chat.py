from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Input

from tui.widgets.answer_panel import AnswerPanel
from tui.widgets.graph_panel import GraphPanel
from tui.widgets.logs_panel import LogsPanel
from tui.widgets.retrieval_panel import RetrievalPanel


class ChatScreen(Screen):
    BINDINGS = [
        ("escape", "pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:

        yield Header()

        with Horizontal(id="main-layout"):
            with Vertical(id="left-column"):
                yield RetrievalPanel()
                yield GraphPanel()

            with Vertical(id="right-column"):
                yield AnswerPanel()
                yield LogsPanel()

        yield Input(
            placeholder="Ask KernelMind about the repository...",
            id="query-input",
        )

        yield Footer()

    def on_input_submitted(self, event: Input.Submitted):

        query = event.value.strip()

        if not query:
            return

        retrieval_panel = self.query_one(RetrievalPanel)
        graph_panel = self.query_one(GraphPanel)
        answer_panel = self.query_one(AnswerPanel)
        logs_panel = self.query_one(LogsPanel)

        retrieval_panel.update_results(
            [
                ("login_access_token", 2.84),
                ("authenticate", 2.43),
                ("verify_password", 2.11),
                ("create_access_token", 1.97),
            ]
        )

        graph_panel.update_trace(
            [
                "login_access_token",
                "authenticate",
                "verify_password",
                "create_access_token",
            ]
        )

        answer_panel.update_answer(
            "The login workflow begins in login_access_token(), "
            "which authenticates the user through authenticate(). "
            "Credentials are validated using verify_password() before "
            "create_access_token() generates the JWT token."
        )

        logs_panel.write_log(f"Query received: {query}")
        logs_panel.write_log("Hybrid retrieval completed")
        logs_panel.write_log("Graph expansion depth = 2")
        logs_panel.write_log("Cross-encoder reranking completed")

        event.input.value = ""
