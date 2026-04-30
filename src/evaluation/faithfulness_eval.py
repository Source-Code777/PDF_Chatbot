def run_faithfulness_evaluation(vectorstore, reranker, llm, eval_llm):
    from src.utils.helpers import generate_query_variations
    from src.llm import generate_answer
    from src.answer_eval_dataset import answer_eval_data
    import os
    import re

    if eval_llm is None:
        print("Evaluation LLM not available")
        return

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    scores = []

    mode = os.getenv("LLM_MODE", "local")

    for item in answer_eval_data:
        query = item["query"]

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

        context = "\n\n---\n\n".join([doc.page_content[:300] for doc in results[:3]])

        model_answer = generate_answer(llm, query, context, [])

        prompt = f"""
You are evaluating answer faithfulness.

Context:
{context}

Answer:
{model_answer}

Instructions:
- Score 1 if the answer is fully supported by the context
- Score 0 if the answer contains hallucinations or unsupported claims

Respond strictly in this format:
Score: 1
or
Score: 0
"""

        response = eval_llm.invoke(prompt)

        match = re.search(r"score\s*[:\-]?\s*(\d)", response.lower())
        score = int(match.group(1)) if match else 0
        scores.append(score)

    print("\nFaithfulness Score:", sum(scores) / len(scores))