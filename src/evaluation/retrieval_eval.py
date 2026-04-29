from utils.helpers import generate_query_variations
from src.eval_dataset import evaluation_data

def is_relevant(doc, keywords):
    content = doc.page_content.lower()
    return any(k in content for k in keywords)

def run_retrieval_evaluation(vectorstore, reranker, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
    scores = []

    for item in evaluation_data:
        query = item["query"]
        keywords = item["keywords"]

        queries = generate_query_variations(llm, query)

        all_docs = []
        for q in queries:
            all_docs.extend(retriever.invoke(q))

        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

        results = reranker.rerank(query, unique_docs, top_k=3)

        score = sum(is_relevant(d, keywords) for d in results) / 3
        scores.append(score)

        print(f"{query} → {score:.2f}")

    print("\nAverage Precision@3:", sum(scores) / len(scores))