from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, documents):
        self.docs = documents
        self.corpus = [doc.page_content.split() for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query, top_k=5):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            list(zip(scores, self.docs)),
            key=lambda x: x[0],
            reverse=True
        )

        return [doc for _, doc in ranked[:top_k]]