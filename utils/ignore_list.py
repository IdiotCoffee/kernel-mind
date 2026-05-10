"""Utility functions for managing the ignore list and configuration."""

import os

from dotenv import load_dotenv

load_dotenv()

IGNORE_DIRS = {
    "tests",
    "__pycache__",
    ".venv",
    "venv",
    ".git",
    "build",
    "dist",
}

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME") or "github-companion"
