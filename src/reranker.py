import os


class Reranker:
    def __init__(self):
        self.mode = os.getenv("RERANK_MODE", "none")

        if self.mode == "local":
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        else:
            self.model = None

    def rerank(self, query, docs, top_k=3):
        if not docs:
            return []

        docs = docs[:8]

        # -------- LOCAL MODE (accurate) --------
        if self.mode == "local" and self.model is not None:
            pairs = [(query, d.page_content[:300]) for d in docs]
            scores = self.model.predict(pairs)

            scored_docs = list(zip(scores, docs))
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            return [doc for _, doc in scored_docs[:top_k]]

        # -------- API MODE (fast fallback) --------
        else:
            # simple heuristic ranking (already prefiltered in run_rag)
            return docs[:top_k]