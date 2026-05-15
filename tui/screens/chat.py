from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Input

from routing.classifier import QueryClassifier
from routing.executor import QueryExecutor
from tui.widgets.answer_panel import AnswerPanel
from tui.widgets.graph_panel import GraphPanel
from tui.widgets.logs_panel import LogsPanel
from tui.widgets.retrieval_panel import RetrievalPanel


class ChatScreen(Screen):
    BINDINGS = [
        ("escape", "pop_screen", "Back"),
    ]

    def __init__(self, runtime, provider):

        super().__init__()

        self.runtime = runtime

        self.provider = provider

        self.classifier = QueryClassifier()

        self.executor = QueryExecutor(
            runtime=runtime,
            provider=provider,
        )

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

    async def on_input_submitted(self, event: Input.Submitted):

        query = event.value.strip()

        if not query:
            return

        retrieval_panel = self.query_one(RetrievalPanel)
        graph_panel = self.query_one(GraphPanel)
        answer_panel = self.query_one(AnswerPanel)
        logs_panel = self.query_one(LogsPanel)

        answer_panel.clear_answer()

        mode = self.classifier.classify(query)

        logs_panel.write_log(f"Mode selected: {mode.value}")

        response = self.executor.execute_with_stream(
            query=query,
            mode=mode,
        )

        # ============================================
        # Retrieval Results
        # ============================================

        formatted_results = []

        for item in response["results"][:10]:
            formatted_results.append(
                (
                    item["fqn"].split(".")[-1],
                    round(
                        item.get(
                            "final_score",
                            item.get("score", 0),
                        ),
                        4,
                    ),
                )
            )

        retrieval_panel.update_results(formatted_results)

        # ============================================
        # Graph Trace
        # ============================================

        graph_panel.update_trace(response["trace"])

        # ============================================
        # Confidence
        # ============================================

        confidence = response["confidence"]

        logs_panel.write_log(
            f"Confidence: {confidence['label']} ({confidence['score']})"
        )

        # ============================================
        # Stream Answer
        # ============================================

        for token in response["stream"]:
            answer_panel.stream_token(token)

        event.input.value = ""
