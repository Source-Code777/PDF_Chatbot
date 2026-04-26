from loader import load_pdf
from splitter import split_documents
from llm import get_eval_llm
from vectorstore import create_vectorstore
from llm import get_llm, generate_answer
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from reranker import CrossEncoderReranker
from evaluation import precision_at_k
from eval_dataset import evaluation_data
from answer_evaluator import run_answer_evaluation
from faithfulness_evaluator import run_faithfulness_evaluation
import os

DB_DIR = "db"

def load_existing_vectorstore_data():
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding
    )
    return vectorstore

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

def run_evaluation(vectorstore, reranker, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    scores = []

    for item in evaluation_data:
        query = item["query"]
        keywords = item["keywords"]

        queries = generate_query_variations(llm, query)

        all_docs = []
        for q in queries:
            docs = retriever.invoke(q)
            all_docs.extend(docs)

        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

        results = reranker.rerank(query, unique_docs, top_k=3)

        score = precision_at_k(results, keywords, k=3)
        scores.append(score)

        print(f"\nQuery: {query}")
        print(f"Precision@3: {score:.2f}")

    avg_score = sum(scores) / len(scores)
    print(f"\nAverage Precision@3: {avg_score:.2f}")

if __name__ == "__main__":
    path = r"C:\Users\aasim\OneDrive\Desktop\Notes\NLP\cs224n_winter2023_lecture1_notes_draft.pdf"

    if os.path.exists(DB_DIR):
        print("Loading existing vectorstore database")
        vectorstore = load_existing_vectorstore_data()
    else:
        print("Creating new vectorstore database")
        documents = load_pdf(path)
        chunks = split_documents(documents)
        vectorstore = create_vectorstore(chunks)

    llm = get_llm()
    eval_llm=get_eval_llm()
    chat_history = []
    reranker = CrossEncoderReranker()
    run_evaluation(vectorstore, reranker, llm)
    print("\n\n Running Answer-Level Evaluation...\n")
    run_answer_evaluation(vectorstore, reranker, llm)
    print("\n\n Running Faithfulness Evaluation...\n")
    run_faithfulness_evaluation(vectorstore, reranker, llm, eval_llm)

    while True:
        query = input("\nAsk: ")
        if query.lower() in ["exit", "quit"]:
            break

        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        queries = generate_query_variations(llm, query)

        all_docs = []
        for q in queries:
            docs = retriever.invoke(q)
            all_docs.extend(docs)

        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        results = reranker.rerank(query, unique_docs, top_k=3)

        context = "\n\n---\n\n".join([doc.page_content for doc in results])
        answer = generate_answer(llm, query, context, chat_history)

        print("\n---Final Answer---\n")
        print(answer)

        chat_history.append(("user", query))
        chat_history.append(("assistant", answer))
        chat_history = chat_history[-6:]