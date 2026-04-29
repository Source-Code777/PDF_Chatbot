from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate


def get_llm():
    return Ollama(
        model="mistral:7b-instruct",
        base_url="http://host.docker.internal:11434",
        temperature=0.0
    )


def get_eval_llm():
    return Ollama(
        model="mistral:7b-instruct",
        base_url="http://host.docker.internal:11434",
        temperature=0.0
    )


def format_chat_history(chat_history):
    history_text = ""
    for role, msg in chat_history:
        if role == "user":
            history_text += f"User: {msg}\n"
        else:
            history_text += f"Assistant: {msg}\n"
    return history_text


def generate_answer(llm, query, context, chat_history):
    if not context or not context.strip():
        return "I don't know based on the provided document."

    history_text = format_chat_history(chat_history)

    prompt_template = PromptTemplate.from_template("""
    You are a helpful AI assistant.

    RULES:
    - Use the provided context to answer the question.
    - Try your best to answer using the context, even if the information is partial.
    - If the context contains related information, form a reasonable answer from it.
    - Only say "I don't know based on the provided document" if the context is completely unrelated.

    ANSWER STYLE:
    - Explain clearly in 2–4 sentences.
    - Use key terms from the context.
    - Be direct and simple.

    Conversation History (for understanding follow-ups only):
    {history}

    Context:
    {context}

    Question:
    {query}

    Answer:
    """)

    final_prompt = prompt_template.format(
        context=context,
        query=query,
        history=history_text
    )

    try:
        response = llm.invoke(final_prompt)
        answer = response.strip()

        if not answer:
            return "I don't know based on the provided document."

        if "i don't know" in answer.lower():
            return "I don't know based on the provided document."

        return answer

    except:
        return "I don't know based on the provided document."