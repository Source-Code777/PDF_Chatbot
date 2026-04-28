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

    variations.append(query)
    return list(set(variations))

from textblob import TextBlob
def correct_spelling(query):
    try:
        blob = TextBlob(query)
        return str(blob.correct())
    except:
        return query

def clean_query(query):
    return " ".join(query.strip().lower().split())

def enrich_query(query, chat_history):
    query = correct_spelling(query)
    query = " ".join(query.strip().lower().split())

    if not chat_history:
        return query

    pronouns = {"it", "this", "that", "they", "them", "its", "those"}
    words = set(query.split())

    if not words.intersection(pronouns):
        return query

    if len(query.split()) > 6:
        return query

    for role, msg in reversed(chat_history):
        if role == "user":
            last = correct_spelling(msg)
            last = " ".join(last.strip().lower().split())
            return last + " " + query

    return query