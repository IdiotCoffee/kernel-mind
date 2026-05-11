import re

from routing.modes import QueryMode


class QueryClassifier:
    """
    Lightweight deterministic router.

    IMPORTANT:
    This class is intentionally simple.

    Later:
    - replace internals with LLM classification
    - keep same external interface
    """

    def __init__(self):

        self.chat_patterns = [
            r"^hi$",
            r"^hello$",
            r"^hey$",
            r"^yo$",
            r"^sup$",
        ]

        self.workflow_patterns = [
            r"how does",
            r"workflow",
            r"flow",
            r"trace",
            r"walk me through",
            r"what happens when",
        ]

        self.symbol_patterns = [
            r"where is",
            r"which function",
            r"which file",
            r"defined",
            r"implemented",
        ]

        self.architecture_patterns = [
            r"architecture",
            r"design",
            r"structure",
            r"organized",
            r"modules",
            r"components",
        ]

        self.existence_patterns = [
            r"does this repo support",
            r"is there support for",
            r"does the repository contain",
            r"is .* implemented",
        ]

    # =====================================================
    # Pattern Matching
    # =====================================================

    def matches_any(
        self,
        text: str,
        patterns,
    ) -> bool:

        for pattern in patterns:
            if re.search(
                pattern,
                text,
                re.IGNORECASE,
            ):
                return True

        return False

    # =====================================================
    # Classification
    # =====================================================

    def classify(
        self,
        query: str,
    ) -> QueryMode:

        normalized = query.strip().lower()

        # ---------------------------------------------
        # CHAT
        # ---------------------------------------------

        if self.matches_any(
            normalized,
            self.chat_patterns,
        ):
            return QueryMode.CHAT

        # ---------------------------------------------
        # WORKFLOW
        # ---------------------------------------------

        if self.matches_any(
            normalized,
            self.workflow_patterns,
        ):
            return QueryMode.WORKFLOW

        # ---------------------------------------------
        # SYMBOL LOOKUP
        # ---------------------------------------------

        if self.matches_any(
            normalized,
            self.symbol_patterns,
        ):
            return QueryMode.SYMBOL_LOOKUP

        # ---------------------------------------------
        # ARCHITECTURE
        # ---------------------------------------------

        if self.matches_any(
            normalized,
            self.architecture_patterns,
        ):
            return QueryMode.ARCHITECTURE

        # ---------------------------------------------
        # EXISTENCE CHECK
        # ---------------------------------------------

        if self.matches_any(
            normalized,
            self.existence_patterns,
        ):
            return QueryMode.EXISTENCE_CHECK

        # ---------------------------------------------
        # GENERAL FALLBACK
        # ---------------------------------------------

        return QueryMode.GENERAL_QA
