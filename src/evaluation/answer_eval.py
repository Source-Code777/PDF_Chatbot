def run_answer_evaluation(vectorstore, reranker, llm):
    from src.answer_eval_dataset import answer_eval_data
    from src.answer_eval_prompt import EVAL_PROMPT
    from src.llm import generate_answer, get_eval_llm
    from utils.helpers import generate_query_variations
    import os
    import re

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    scores = []

    eval_llm = get_eval_llm()
    if eval_llm is None:
        print("Evaluation LLM not available")
        return

    mode = os.getenv("LLM_MODE", "local")

    for item in answer_eval_data:
        query = item["query"]
        ground_truth = item["ground_truth"]

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

        eval_prompt = EVAL_PROMPT.format(
            query=query,
            ground_truth=ground_truth,
            model_answer=model_answer
        )

        eval_response = eval_llm.invoke(eval_prompt)

        print(f"\nQuery: {query}")
        print(eval_response)

        match = re.search(r"score\s*:\s*(\d)", eval_response.lower())
        score = int(match.group(1)) if match else 0
        scores.append(score)

    print("\nFinal Accuracy:", sum(scores) / len(scores))