import os

from dotenv import load_dotenv
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Static

from generation.providers.sarvam_provider import SarvamProvider
from indexing.repository_runtime import RepositoryRuntime
from tui.screens.chat import ChatScreen

load_dotenv()
ASCII_ART = r"""
██╗  ██╗███╗   ███╗
██║ ██╔╝████╗ ████║
█████╔╝ ██╔████╔██║
██╔═██╗ ██║╚██╔╝██║
██║  ██╗██║ ╚═╝ ██║
╚═╝  ╚═╝╚═╝     ╚═╝
"""


class HomeScreen(Screen):
    def compose(self) -> ComposeResult:

        with Horizontal(id="hero-layout"):
            # ============================================
            # ASCII SIDE
            # ============================================

            with Vertical(id="ascii-panel"):
                yield Static(
                    f"[bold orange1]{ASCII_ART}[/bold orange1]",
                    markup=True,
                    id="ascii-art",
                )

            # ============================================
            # CONTENT SIDE
            # ============================================

            with Vertical(id="hero-content"):
                yield Static(
                    "[bold orange1]KernelMind v2[/bold orange1]",
                    markup=True,
                    id="hero-title",
                )

                yield Static(
                    "Graph-Aware Repository Intelligence",
                    id="hero-subtitle",
                )

                yield Static(
                    "Hybrid retrieval, graph expansion, reranking, "
                    "workflow reconstruction, and grounded repository reasoning.",
                    id="hero-description",
                )

                yield Button(
                    "Explore",
                    variant="primary",
                    id="launch-chat",
                )

    def on_button_pressed(self, event: Button.Pressed):

        if event.button.id == "launch-chat":
            repo_id = "full-stack-fastapi-template"

            runtime = RepositoryRuntime.load(
                repo_id=repo_id,
                device="cuda",
            )

            provider = SarvamProvider(api_key=os.getenv("SARVAM_API_KEY", ""))

            self.app.push_screen(
                ChatScreen(
                    runtime=runtime,
                    provider=provider,
                )
            )
