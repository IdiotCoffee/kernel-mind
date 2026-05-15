import os

from dotenv import load_dotenv
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Static

from generation.providers.sarvam_provider import SarvamProvider
from indexing.process_repository import (
    process_repository,
)
from indexing.runtime_builder import (
    build_runtime_from_repo,
)
from tui.screens.chat import ChatScreen

load_dotenv()

DEFAULT_REPO = "https://github.com/fastapi/full-stack-fastapi-template"

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
                    "Hybrid retrieval, graph expansion, "
                    "reranking, workflow reconstruction, "
                    "and grounded repository reasoning.",
                    id="hero-description",
                )

                yield Input(
                    placeholder=("Repository URL (leave empty for demo repo)"),
                    id="repo-input",
                )

                yield Button(
                    "Explore",
                    variant="default",
                    id="launch-chat",
                )

    def on_button_pressed(self, event: Button.Pressed):

        if event.button.id != "launch-chat":
            return

        # ============================================
        # Repo Input
        # ============================================

        repo_input = self.query_one(
            "#repo-input",
            Input,
        )

        repo_url = repo_input.value.strip()

        # ============================================
        # Default Repo
        # ============================================

        if not repo_url:
            repo_url = DEFAULT_REPO

        # ============================================
        # Notifications
        # ============================================

        self.notify(
            "Processing repository...",
            timeout=3,
        )

        # ============================================
        # Process Repository
        # ============================================

        repo_data = process_repository(repo_url)

        # ============================================
        # Build Runtime
        # ============================================

        runtime = build_runtime_from_repo(
            repo_id=repo_data["repo_id"],
            chunks=repo_data["chunks"],
            graph=repo_data["graph"],
            device="cuda",
        )

        # ============================================
        # Provider
        # ============================================

        provider = SarvamProvider(
            api_key=os.getenv(
                "SARVAM_API_KEY",
                "",
            )
        )

        # ============================================
        # Launch Chat
        # ============================================

        self.app.push_screen(
            ChatScreen(
                runtime=runtime,
                provider=provider,
            )
        )
