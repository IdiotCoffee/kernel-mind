from textual.containers import Vertical, VerticalScroll
from textual.widgets import Markdown, Static


class AnswerPanel(Vertical):
    def compose(self):

        yield Static(
            "[bold orange1]Grounded Answer[/bold orange1]",
            markup=True,
            classes="panel-title",
        )

        self.current_text = ""

        self.answer_widget = Markdown(
            "",
            # markup=True,
            id="answer-text",
        )

        with VerticalScroll():
            yield self.answer_widget

    def clear_answer(self):

        self.current_text = ""

        self.answer_widget.update("")

    def stream_token(self, token: str):

        self.current_text += str(token)

        self.answer_widget.update(self.current_text)

    def update_answer(self, answer: str):

        self.current_text = answer

        self.answer_widget.update(self.current_text)
