from transformers import pipeline
from langchain_core.prompts import PromptTemplate

def get_llm():
    generator = pipeline(
        task="text-generation",
        model="HuggingFaceTB/SmolLM2-1.7B",
        max_new_tokens=100,
        do_sample=False
    )
    return generator


def generate_answer(llm, query, context):
    prompt_template = PromptTemplate.from_template("""
Answer the question based on the context below.
If you don't know the answer, just say you don't know.

Context:
{context}

Question:
{query}

Answer:
""")

    final_prompt = prompt_template.format(
        context=context,
        query=query
    )

    response = llm(final_prompt)
    return response[0]['generated_text']