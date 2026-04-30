import os
from groq import Groq
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate


class GroqLLM:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def invoke(self, prompt):
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.2,
                max_tokens=200
            )
            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"API Error: {str(e)}")


def get_llm():
    mode = os.getenv("LLM_MODE", "local")

    if mode == "local":
        return Ollama(
            model="mistral:7b-instruct",
            base_url="http://host.docker.internal:11434",
            temperature=0.0
        )

    elif mode == "api":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        return GroqLLM(api_key)


def get_eval_llm():
    return get_llm()


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

    if llm is None:
        return "Missing Groq API key. Please add it in API mode."

    history_text = format_chat_history(chat_history)

    prompt_template = PromptTemplate.from_template("""
You are a helpful AI assistant.

RULES:
- Use the provided context to answer the question.
- Try your best to answer using the context.
- Only say "I don't know based on the provided document" if completely unrelated.

Conversation History:
{history}

Context:
{context}

Question:
{query}

Answer:
""")

    final_prompt = prompt_template.format(
        context=context[:2000],
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

    except Exception as e:
        return str(e)