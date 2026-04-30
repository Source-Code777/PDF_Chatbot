def run_rag(query, vectorstore, llm, reranker, bm25, chat_history=None):
    from src.utils.helpers import generate_query_variations, enrich_query
    from src.llm import generate_answer
    import os
    import re

    def tokenize(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return set(text.split())

    chat_history = chat_history if chat_history is not None else []

    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    query = enrich_query(query, chat_history)

    mode = os.getenv("LLM_MODE", "local")

    if mode == "local":
        try:
            queries = generate_query_variations(llm, query)[:5]
        except:
            queries = [query]
    else:
        queries = [query]

    if query not in queries:
        queries.append(query)

    all_docs = []
    seen = set()

    for q in queries:
        dense_docs = retriever.invoke(q)
        sparse_docs = bm25.retrieve(q, top_k=5)

        for doc in dense_docs + sparse_docs:
            key = doc.page_content[:150]
            if key not in seen:
                all_docs.append(doc)
                seen.add(key)

    if not all_docs:
        return "I don't know based on the provided document.", "", []

    query_tokens = tokenize(query)

    scored = []
    for doc in all_docs:
        doc_tokens = tokenize(doc.page_content)
        score = len(query_tokens & doc_tokens)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    prefiltered = [doc for _, doc in scored[:12]]

    results = reranker.rerank(query, prefiltered, top_k=3)

    if not results:
        results = prefiltered[:3]

    context = "\n\n---\n\n".join([doc.page_content[:300] for doc in results[:3]])

    answer = generate_answer(llm, query, context, chat_history)

    return answer, context, results