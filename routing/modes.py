from enum import Enum


class QueryMode(str, Enum):
    CHAT = "chat"

    WORKFLOW = "workflow"

    SYMBOL_LOOKUP = "symbol_lookup"

    ARCHITECTURE = "architecture"

    EXISTENCE_CHECK = "existence_check"

    GENERAL_QA = "general_qa"

    UNKNOWN = "unknown"
