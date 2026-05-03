TYPE_BOOST = {
    "function": 0.2,
    "method": 0.18,
    "class": 0.1,
}


def combine_scores(bm25_scores, metas):
    scores = []

    for i, base in enumerate(bm25_scores):
        boost = TYPE_BOOST.get(metas[i].get("type"), 0)
        scores.append(base + boost)

    return scores
