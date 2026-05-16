import asyncio
import subprocess

from textual.containers import Vertical, VerticalScroll
from textual.widgets import Markdown, Static


class AnswerPanel(Vertical):
    repo_root = ""

    def compose(self):

        yield Static(
            "[bold orange1]Grounded Answer[/bold orange1]",
            markup=True,
            classes="panel-title",
        )

        self.current_text = ""

        self.loading = False

        self.answer_widget = Markdown(
            "",
            id="answer-text",
        )

        with VerticalScroll():
            yield self.answer_widget

    # =====================================================
    # Loading Animation
    # =====================================================

    async def loading_animation(self):

        dots = ["⠁", "⠂", "⠄", "⠂"]

        idx = 0

        while self.loading:
            dot = dots[idx % len(dots)]

            self.answer_widget.update(
                f"\n\n[bold orange1]KernelMind reasoning {dot}[/bold orange1]"
            )

            idx += 1

            await asyncio.sleep(0.25)

    def start_loading(self):

        self.loading = True

        asyncio.create_task(self.loading_animation())

    def stop_loading(self):

        self.loading = False

    # =====================================================
    # Answer Operations
    # =====================================================

    def clear_answer(self):

        self.current_text = ""

        self.answer_widget.update("")

    def stream_token(self, token: str):

        self.current_text += str(token)

        self.answer_widget.update(self.current_text)

    def update_answer(self, answer: str):

        self.current_text = answer

        self.answer_widget.update(self.current_text)

    # =====================================================
    # Source Navigation
    # =====================================================

    def on_markdown_link_clicked(
        self,
        event: Markdown.LinkClicked,
    ):

        href = event.href or ""

        # -----------------------------------------
        # Only handle source:// links
        # -----------------------------------------

        if not href.startswith("source://"):
            return
        event.stop()

        cleaned = href.replace(
            "source://",
            "",
        )

        # -----------------------------------------
        # Parse line number
        # -----------------------------------------

        if "#L" in cleaned:
            relative_path, line_str = cleaned.split("#L")

            line_number = int(line_str)

        else:
            relative_path = cleaned

            line_number = 1

        # -----------------------------------------
        # Repo root
        # -----------------------------------------

        repo_root = self.repo_root

        full_path = f"{repo_root}/{relative_path}"

        # -----------------------------------------
        # Open Zed
        # -----------------------------------------

        subprocess.Popen(
            [
                "zeditor",
                f"{full_path}:{line_number}",
            ]
        )
