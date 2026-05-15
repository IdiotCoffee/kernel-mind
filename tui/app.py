from textual.app import App

from tui.screens.home import HomeScreen


class KernelMindApp(App):
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]

    def on_mount(self):
        self.push_screen(HomeScreen())


if __name__ == "__main__":
    app = KernelMindApp()
    app.run()
