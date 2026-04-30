from utils.helpers import generate_query_variations
from src.eval_dataset import evaluation_data
import os
import re


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return set(text.split())


def is_relevant(doc, keywords):
    content_tokens = tokenize(doc.page_content)
    keyword_tokens = set(k.lower() for k in keywords)
    return len(content_tokens & keyword_tokens) > 0


def run_retrieval_evaluation(vectorstore, reranker, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
    scores = []

    mode = os.getenv("LLM_MODE", "local")

    for item in evaluation_data:
        query = item["query"]
        keywords = item["keywords"]

        if mode == "local":
            try:
                queries = generate_query_variations(llm, query)
            except:
                queries = [query]
        else:
            queries = [query]

        all_docs = []
        for q in queries:
            all_docs.extend(retriever.invoke(q))

        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

        results = reranker.rerank(query, unique_docs, top_k=3)

        if not results:
            results = unique_docs[:3]

        k = len(results) if results else 1
        score = sum(is_relevant(d, keywords) for d in results) / k
        scores.append(score)

        print(f"\nQuery: {query}")
        print(f"Score: {score:.2f}")
        print("Top Docs Preview:")
        for d in results:
            print("-", d.page_content[:100])

    print("\nAverage Precision@3:", sum(scores) / len(scores))