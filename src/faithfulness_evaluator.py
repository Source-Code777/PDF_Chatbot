from llm import generate_answer

FAITHFULNESS_PROMPT = """
You are an evaluator.

Context:
{context}

Model Answer:
{model_answer}

Task:
Determine whether the model answer is fully supported by the provided context.

Evaluation Rules:
- The answer must be grounded ONLY in the context
- If the answer contains information NOT present in the context → Score = 0
- If all parts of the answer are supported by the context → Score = 1

Important:
- Be strict
- Do not assume missing information
- Do not use outside knowledge

Respond ONLY in this format:
Score: 0 or 1
Reason: <short explanation>
"""


def run_faithfulness_evaluation(vectorstore, reranker, llm, eval_llm):
    from main import generate_query_variations
    from answer_eval_dataset import answer_eval_data

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    scores = []

    for item in answer_eval_data:
        query = item["query"]

        #Same pipeline
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

        #Faithfulness check
        eval_prompt = FAITHFULNESS_PROMPT.format(
            context=context,
            model_answer=model_answer
        )

        eval_response = eval_llm.invoke(eval_prompt)

        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print("\nModel Answer:\n", model_answer)
        print("\nFaithfulness Evaluation:\n", eval_response)

        if "Score: 1" in eval_response:
            scores.append(1)
        else:
            scores.append(0)

    final_score = sum(scores) / len(scores)

    print("\n" + "=" * 60)
    print(f"Final Faithfulness Score: {final_score:.2f}")

    return final_score