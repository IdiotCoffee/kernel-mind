import asyncio

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
        # answer_panel.repo_root = self.runtime.repo_path
        answer_panel.repo_root = (
            "/home/idiotcoffee/Desktop/kernel-mind-v2/"
            f"kernel-mind/repos/{self.runtime.repo_id}"
        )
        logs_panel = self.query_one(LogsPanel)
        # logs_panel.write_log(str(self.runtime.__dict__))

        answer_panel.clear_answer()
        answer_panel.start_loading()

        # allow UI repaint
        await asyncio.sleep(0.05)

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
        # await asyncio.sleep(0.05)
        answer_panel.stop_loading()
        full_answer = ""

        for token in response["stream"]:
            full_answer += token

            answer_panel.stream_token(token)

        # ============================================
        # Append deterministic citations
        # ============================================

        citations = []

        seen = set()

        for item in response["results"][:8]:
            chunk = self.runtime.chunk_lookup.get(item["fqn"])

            if not chunk:
                continue

            key = (
                chunk.file_path,
                chunk.start_line,
            )

            if key in seen:
                continue

            seen.add(key)

            citations.append(
                (
                    f"- "
                    f"[{chunk.file_path}]"
                    f"(source://{chunk.file_path}"
                    f"#L{chunk.start_line})"
                )
            )

        if citations:
            citation_block = "\n\n## Sources\n\n" + "\n".join(citations)

            answer_panel.stream_token(citation_block)

        event.input.value = ""
