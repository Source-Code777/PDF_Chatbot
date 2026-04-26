from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate


def get_llm():
    generator = Ollama(
        model="llama3.2:1b",
        temperature=0.0
    )
    return generator

def get_eval_llm():
    return Ollama(
        model="mistral:7b-instruct",
        temperature=0.0
    )


def format_chat_history(chat_history):
    history_text = ""
    for role, msg in chat_history:
        if role=="user":
            history_text+=f"User: {msg}\n"
        else:
            history_text+=f"Assistant: {msg}\n"
    return history_text

def is_context_relevant(query, context):
    query_words = [w for w in query.lower().split() if len(w) > 3]
    context_lower = context.lower()

    match_count = sum(1 for w in query_words if w in context_lower)

    return match_count >= 1

def generate_answer(llm, query, context, chat_history):
    history_text=format_chat_history(chat_history)
    prompt_template = PromptTemplate.from_template("""
    You are a helpful AI assistant.

    IMPORTANT RULES:
    - Answer ONLY using the given context
    - Do NOT use outside knowledge
    - If the answer is clearly present in the context → answer normally
    - ONLY say "I don't know based on the provided document" IF no relevant information exists
    - Do NOT say both an answer and "I don't know"
    - Do NOT introduce new concepts not present in the context

    - Give a clear and slightly detailed explanation (2-3 sentences)
    - Include key terms and concepts if available
    - Use ONLY information from the context
    - Do NOT add assumptions or extra explanations not present in context

    - At the end, add: "(Based on retrieved context)"

    Conversation History:
    {history}

    Context:
    {context}

    Question:
    {query}

    Answer:
    """)
    if not is_context_relevant(query, context):
        return "I don't know based on the provided document."

    final_prompt = prompt_template.format(
        context=context,
        query=query,
        history=history_text
    )

    response = llm.invoke(final_prompt)

    answer = response.strip()
    return answer