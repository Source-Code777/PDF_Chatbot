from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, docs, top_k=3):
        if not docs:
            return []

        docs = docs[:8]
        pairs = [(query, doc.page_content[:300]) for doc in docs]

        scores = self.model.predict(pairs, batch_size=8)

        scored_docs = list(zip(scores, docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored_docs[:top_k]]