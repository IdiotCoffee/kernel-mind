import re
from typing import List

from rank_bm25 import BM25Okapi

from db.models import CodeChunk

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "and",
    "or",
    "if",
    "else",
    "return",
    "true",
    "false",
    "none",
}


def split_camel_case(text: str) -> str:

    return re.sub(
        r"([a-z])([A-Z])",
        r"\1 \2",
        text,
    )


def tokenize(text: str) -> List[str]:
    """
    Code-aware tokenizer.

    Handles:
    - camelCase
    - snake_case
    - dotted.module.paths
    - symbols
    - stopwords
    """

    text = split_camel_case(text)

    text = text.lower()

    # -----------------------------------
    # Replace separators
    # -----------------------------------

    text = re.sub(
        r"[._/()=:,\[\]{}<>+-]",
        " ",
        text,
    )

    tokens = text.split()

    # -----------------------------------
    # Remove stopwords
    # -----------------------------------

    cleaned = []

    for token in tokens:
        if token in STOPWORDS:
            continue

        if len(token) <= 1:
            continue

        cleaned.append(token)

    return cleaned


class BM25Retriever:
    def __init__(
        self,
        chunks: List[CodeChunk],
    ):
        self.chunks = chunks

        self.documents = []

        # -----------------------------------
        # Build tokenized corpus
        # -----------------------------------

        for chunk in chunks:
            doc = self.build_document(chunk)

            self.documents.append(tokenize(doc))

        # -----------------------------------
        # Initialize BM25
        # -----------------------------------

        self.bm25 = BM25Okapi(self.documents)

    def build_document(
        self,
        chunk: CodeChunk,
    ) -> str:
        """
        Build searchable lexical document.
        """

        parts = [
            chunk.fqn,
            chunk.type,
            chunk.module,
        ]

        # -----------------------------------
        # Add docstring
        # -----------------------------------

        if chunk.docstring:
            parts.append(chunk.docstring)

        # -----------------------------------
        # Add raw code
        # -----------------------------------

        if chunk.code:
            parts.append(chunk.code)

        # -----------------------------------
        # Add calls
        # -----------------------------------

        if chunk.calls:
            parts.extend(chunk.calls)

        # -----------------------------------
        # Add imports
        # -----------------------------------

        if chunk.imports:
            parts.extend(chunk.imports.keys())

            parts.extend(chunk.imports.values())

        return "\n".join(parts)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[dict]:

        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        results = []

        for idx, score in enumerate(scores):
            results.append(
                {
                    "score": float(score),
                    "chunk": self.chunks[idx],
                }
            )

        results.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return results[:top_k]
