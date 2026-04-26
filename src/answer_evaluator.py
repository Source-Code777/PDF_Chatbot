from answer_eval_dataset import answer_eval_data
from answer_eval_prompt import EVAL_PROMPT
from llm import generate_answer
from llm import get_eval_llm


def run_answer_evaluation(vectorstore, reranker, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    scores = []

    from main import generate_query_variations

    for item in answer_eval_data:
        query = item["query"]
        ground_truth = item["ground_truth"]


        queries = generate_query_variations(llm, query)

        all_docs = []
        for q in queries:
            docs = retriever.invoke(q)
            all_docs.extend(docs)

        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())


        results = reranker.rerank(query, unique_docs, top_k=3)

        context = "\n\n---\n\n".join([doc.page_content for doc in results])

        model_answer = generate_answer(
            llm, query, context, chat_history=[]
        )

        eval_prompt = EVAL_PROMPT.format(
            query=query,
            ground_truth=ground_truth,
            model_answer=model_answer
        )
        eval_llm=get_eval_llm()
        eval_response = eval_llm.invoke(eval_prompt)

        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print("\nModel Answer:\n", model_answer)
        print("\nEvaluation:\n", eval_response)

        if "Score: 1" in eval_response:
            scores.append(1)
        else:
            scores.append(0)

    accuracy = sum(scores) / len(scores)

    print("\n" + "=" * 60)
    print(f"Final Answer Accuracy: {accuracy:.2f}")

    return accuracy