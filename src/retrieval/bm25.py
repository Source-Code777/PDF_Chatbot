from rank_bm25 import BM25Okapi
import re
import numpy as np

STOPWORDS = {
    "the", "is", "in", "at", "of", "on", "and", "a", "to", "for",
    "what", "which", "who", "how", "when", "where", "why"
}


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]


class BM25Retriever:
    def __init__(self, documents):
        self.docs = documents
        self.corpus = [tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query, top_k=5):
        tokenized_query = tokenize(query)

        # fallback for empty query
        if not tokenized_query:
            return self.docs[:top_k]

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.docs[i] for i in top_indices]