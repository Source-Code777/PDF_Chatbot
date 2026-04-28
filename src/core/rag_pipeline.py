def run_rag(query, vectorstore, llm, reranker, chat_history=None):
    from utils.helpers import generate_query_variations, enrich_query
    from llm import generate_answer

    chat_history = chat_history if chat_history is not None else []

    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

    query = enrich_query(query, chat_history)

    queries = [query]
    try:
        variations = generate_query_variations(llm, query)
        for v in variations:
            if v not in queries:
                queries.append(v)
            if len(queries) >= 3:
                break
    except:
        pass

    all_docs = []
    for q in queries:
        all_docs.extend(retriever.invoke(q))

    seen = set()
    unique_docs = []
    for doc in all_docs:
        key = doc.page_content[:200]
        if key not in seen:
            unique_docs.append(doc)
            seen.add(key)

    if not unique_docs:
        return "I don't know based on the provided document.", "", []

    prefiltered = unique_docs[:8]

    results = reranker.rerank(query, prefiltered, top_k=3)

    context = "\n\n---\n\n".join([doc.page_content[:300] for doc in results])

    answer = generate_answer(llm, query, context, chat_history)

    return answer, context, results