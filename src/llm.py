from transformers import pipeline
from langchain_core.prompts import PromptTemplate

def get_llm():
    generator = pipeline(
        task="text-generation",
        model="HuggingFaceTB/SmolLM2-1.7B",
        max_new_tokens=200,
        do_sample=False
    )
    return generator

def format_chat_history(chat_history):
    history_text=""
    for rol,msg in enumerate(chat_history):
        if rol=="user":
            history_text+=f"User: {msg}\n"
        else:
            history_text+=f"Assistant: {msg}\n"
        return history_text




def generate_answer(llm, query, context, chat_history):
    history_text=format_chat_history(chat_history)
    prompt_template = PromptTemplate.from_template("""
    You are a helpful AI assistant.

    Your job is to READ the context and EXPLAIN it in a simple and clear way.
    
    Rules:
    - Do NOT copy text directly from the context
    - Always explain in your own words
    - Keep answers short and easy to understand
    - If the question depends on previous conversation, use the history
    
    Conversation History:
    {history}
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer (simple explanation):
    """
    )
    final_prompt = prompt_template.format(
        context=context,
        query=query,
        history=history_text
    )

    response = llm(final_prompt)
    full_text=response[0]['generated_text']
    if final_prompt in full_text:
        answer = full_text.split(final_prompt)[-1].strip()
    else:
        answer = full_text.strip()
    return answer