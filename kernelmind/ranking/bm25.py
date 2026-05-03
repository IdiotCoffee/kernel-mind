import re

from rank_bm25 import BM25Okapi

token_pattern = re.compile(r"\w+")


def tokenize(text):
    return token_pattern.findall((text or "").lower())


def bm25_score(query, docs):
    corpus = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(corpus)

    return bm25.get_scores(tokenize(query))
