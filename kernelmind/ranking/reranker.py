try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


class Reranker:
    def __init__(self):
        self.model = None

        if CrossEncoder:
            try:
                self.model = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda"
                )
            except Exception:
                self.model = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
                )

    def rerank(self, query, docs):
        if not self.model:
            return None

        pairs = [[query, d] for d in docs]
        return self.model.predict(pairs)
