def generate_query_variations(llm, query):
    prompt = f"""
You are an expert in information retrieval.

Given a user question, generate 5 diverse search queries that would help retrieve relevant documents.

Rules:
- Use different wording and terminology
- Expand abbreviations if possible
- Include both short and detailed queries
- Do NOT number them
- One query per line

User question: {query}
"""
    response = llm.invoke(prompt)
    lines = response.strip().split("\n")

    variations = []
    for line in lines:
        clean = line.strip().lstrip("1234567890. ").strip()
        if clean:
            variations.append(clean)

    # ensure original query is always included
    variations.append(query)

    # remove duplicates
    return list(set(variations))


def clean_query(query):
    return " ".join(query.strip().lower().split())


def enrich_query(query, chat_history):
    # normalize query (no spell correction anymore)
    query = clean_query(query)

    if not chat_history:
        return query

    pronouns = {"it", "this", "that", "they", "them", "its", "those"}
    words = set(query.split())

    # if no pronouns → no need to modify
    if not words.intersection(pronouns):
        return query

    # avoid over-expanding long queries
    if len(query.split()) > 6:
        return query

    # attach previous user query for context
    for role, msg in reversed(chat_history):
        if role == "user":
            last = clean_query(msg)
            return last + " " + query

    return query