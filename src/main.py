import os
from collections import deque

from core.rag_pipeline import run_rag
from loader import load_pdf
from splitter import split_documents
from vectorstore import create_vectorstore, load_existing_vectorstore
from llm import get_llm, get_eval_llm
from reranker import CrossEncoderReranker

from evaluation.retrieval_eval import run_retrieval_evaluation
from evaluation.answer_eval import run_answer_evaluation
from evaluation.faithfulness_eval import run_faithfulness_evaluation

DB_DIR = "db"

def main():
    path = r"C:\Users\aasim\OneDrive\Desktop\Notes\NLP\cs224n_winter2023_lecture1_notes_draft.pdf"

    if os.path.exists(DB_DIR):
        print("Loading vectorstore...")
        vectorstore = load_existing_vectorstore()
    else:
        print("Creating vectorstore...")
        docs = load_pdf(path)
        chunks = split_documents(docs)
        vectorstore = create_vectorstore(chunks)

    llm = get_llm()
    eval_llm = get_eval_llm()
    reranker = CrossEncoderReranker()

    print("\nRunning Retrieval Evaluation...\n")
    run_retrieval_evaluation(vectorstore, reranker, llm)

    print("\nRunning Answer Evaluation...\n")
    run_answer_evaluation(vectorstore, reranker, llm)

    print("\nRunning Faithfulness Evaluation...\n")
    run_faithfulness_evaluation(vectorstore, reranker, llm, eval_llm)

    chat_history = deque(maxlen=6)

    print("\n--- Chatbot Ready ---\n")

    while True:
        query = input("Ask: ")

        if query.lower() in ["exit", "quit"]:
            break

        answer, context, docs = run_rag(
            query, vectorstore, llm, reranker, chat_history
        )

        print("\n--- Final Answer ---\n")
        print(answer)

        chat_history.append(("user", query))
        chat_history.append(("assistant", answer))


if __name__ == "__main__":
    main()