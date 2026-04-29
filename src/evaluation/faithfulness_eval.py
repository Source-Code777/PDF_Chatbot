from src.utils.helpers import generate_query_variations
from src.llm import generate_answer
from src.answer_eval_dataset import answer_eval_data

def run_faithfulness_evaluation(vectorstore, reranker, llm, eval_llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    scores = []

    for item in answer_eval_data:
        query = item["query"]

        queries = generate_query_variations(llm, query)

        all_docs = []
        for q in queries:
            all_docs.extend(retriever.invoke(q))

        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

        results = reranker.rerank(query, unique_docs, top_k=3)

        context = "\n\n---\n\n".join([doc.page_content for doc in results])

        model_answer = generate_answer(llm, query, context, [])

        prompt = f"""
Context:
{context}

Answer:
{model_answer}

Score 1 if grounded else 0
"""

        response = eval_llm.invoke(prompt)

        scores.append(1 if "1" in response else 0)

    print("\nFaithfulness Score:", sum(scores) / len(scores))