import re
from typing import List


def tokenize(text: str) -> List[str]:
    """
    Repository-aware tokenizer.

    Handles:
    - snake_case
    - dotted paths
    - camelCase
    - kebab-case
    """

    # -----------------------------------
    # Split camelCase
    # -----------------------------------

    text = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        text,
    )

    # -----------------------------------
    # Replace separators
    # -----------------------------------

    separators = [
        ".",
        "_",
        "-",
        "/",
        "\\",
        ":",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        ",",
    ]

    for sep in separators:
        text = text.replace(sep, " ")

    # -----------------------------------
    # Normalize whitespace
    # -----------------------------------

    text = re.sub(r"\s+", " ", text)

    # -----------------------------------
    # Final tokens
    # -----------------------------------

    return [token.lower() for token in text.strip().split() if token.strip()]
