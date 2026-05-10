from typing import List

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda",
    ):
        self.model = CrossEncoder(
            model_name,
            device=device,
        )

    def build_pair(
        self,
        query: str,
        chunk,
    ) -> List[str]:

        parts = [
            f"FQN: {chunk.fqn}",
            f"TYPE: {chunk.type}",
        ]

        if chunk.docstring:
            parts.append(chunk.docstring)

        if chunk.code:
            parts.append(chunk.code[:1500])

        text = "\n".join(parts)

        return [query, text]

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        chunk_lookup: dict,
        top_k: int = 10,
    ) -> List[dict]:
        print("\ncross-encoder in progress...")
        pairs = []

        valid_candidates = []

        # -----------------------------------
        # Build query/chunk pairs
        # -----------------------------------

        for item in candidates:
            chunk = chunk_lookup.get(item["fqn"])

            if not chunk:
                continue

            pair = self.build_pair(
                query=query,
                chunk=chunk,
            )

            pairs.append(pair)

            valid_candidates.append((item, chunk))

        # -----------------------------------
        # Cross-encoder scoring
        # -----------------------------------

        scores = self.model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )

        reranked = []

        # -----------------------------------
        # Normalize cross-encoder scores
        # -----------------------------------

        min_score = float(min(scores))
        max_score = float(max(scores))

        score_range = float(max_score - min_score)

        reranked = []

        for (item, chunk), score in zip(valid_candidates, scores):
            # -----------------------------------
            # Min-max normalization
            # -----------------------------------

            if score_range == 0:
                normalized_cross_score = 0.5

            else:
                normalized_cross_score = float((float(score) - min_score) / score_range)
            # -----------------------------------
            # Final blended score
            # -----------------------------------

            # final_score = item["score"] * 0.4 + normalized_cross * 0.6
            final_score = item["score"] * 0.4 + normalized_cross_score * 0.6
            reranked.append(
                {
                    **item,
                    "cross_score": float(score),
                    "normalized_cross_score": round(
                        normalized_cross_score,
                        4,
                    ),
                    "final_score": round(
                        final_score,
                        4,
                    ),
                }
            )

        # -----------------------------------
        # Final sort
        # -----------------------------------

        reranked.sort(
            key=lambda x: x["final_score"],
            reverse=True,
        )
        print("\nRERANK DEBUG:\n")

        for item in reranked[:10]:
            print(
                item["fqn"],
                "| graph =",
                round(item["score"], 4),
                "| cross =",
                round(item["cross_score"], 4),
                "| norm =",
                round(item["normalized_cross_score"], 4),
                "| final =",
                round(item["final_score"], 4),
            )
        return reranked[:top_k]
