from answer_eval_dataset import answer_eval_data
from answer_eval_prompt import EVAL_PROMPT
from llm import generate_answer, get_eval_llm
from utils.helpers import generate_query_variations

def run_answer_evaluation(vectorstore, reranker, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    scores = []

    eval_llm = get_eval_llm()

    for item in answer_eval_data:
        query = item["query"]
        ground_truth = item["ground_truth"]

        queries = generate_query_variations(llm, query)

        all_docs = []
        for q in queries:
            all_docs.extend(retriever.invoke(q))

        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

        results = reranker.rerank(query, unique_docs, top_k=3)

        context = "\n\n---\n\n".join([doc.page_content for doc in results])

        model_answer = generate_answer(llm, query, context, [])

        eval_prompt = EVAL_PROMPT.format(
            query=query,
            ground_truth=ground_truth,
            model_answer=model_answer
        )

        eval_response = eval_llm.invoke(eval_prompt)

        print(f"\nQuery: {query}")
        print(eval_response)

        scores.append(1 if "Score: 1" in eval_response else 0)

    print("\nFinal Accuracy:", sum(scores) / len(scores))