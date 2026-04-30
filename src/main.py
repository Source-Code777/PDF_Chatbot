import os
from collections import deque

from retrieval.bm25 import BM25Retriever

from src.core.rag_pipeline import run_rag
from src.loader import load_pdf
from src.splitter import split_documents
from src.vectorstore import create_vectorstore, load_existing_vectorstore
from src.llm import get_llm, get_eval_llm
from src.reranker import Reranker

from src.evaluation.retrieval_eval import run_retrieval_evaluation
from src.evaluation.answer_eval import run_answer_evaluation
from src.evaluation.faithfulness_eval import run_faithfulness_evaluation

DATA_DIR = "data"


def main():
    # ---------- MODE SELECTION ----------
    mode = input("Choose mode (local/api): ").strip().lower()

    if mode == "api":
        os.environ["LLM_MODE"] = "api"
        os.environ["RERANK_MODE"] = "api"

        groq_key = input("Enter Groq API Key: ").strip()
        os.environ["GROQ_API_KEY"] = groq_key
    else:
        os.environ["LLM_MODE"] = "local"
        os.environ["RERANK_MODE"] = "local"

    # ---------- PDF PATH ----------
    path = r"C:\Users\aasim\OneDrive\Desktop\Notes\NLP\cs224n_winter2023_lecture1_notes_draft.pdf"

    pdf_name = os.path.basename(path).replace(".pdf", "")
    pdf_dir = os.path.join(DATA_DIR, pdf_name)
    db_dir = os.path.join(pdf_dir, "vectordb")

    os.makedirs(pdf_dir, exist_ok=True)

    docs = load_pdf(path)
    chunks = split_documents(docs)

    # ---------- VECTORSTORE ----------
    if os.path.exists(db_dir):
        print("Loading vectorstore...")
        vectorstore = load_existing_vectorstore(db_dir)
    else:
        print("Creating vectorstore...")
        vectorstore = create_vectorstore(chunks, db_dir)

    bm25 = BM25Retriever(chunks)

    # ---------- MODELS ----------
    llm = get_llm()
    eval_llm = get_eval_llm()
    reranker = Reranker()

    # ---------- EVALUATION ----------
    if mode == "local":
        print("\nRunning Retrieval Evaluation...\n")
        run_retrieval_evaluation(vectorstore, reranker, llm)

        print("\nRunning Answer Evaluation...\n")
        run_answer_evaluation(vectorstore, reranker, llm)

        if eval_llm is not None:
            print("\nRunning Faithfulness Evaluation...\n")
            run_faithfulness_evaluation(vectorstore, reranker, llm, eval_llm)
        else:
            print("\nSkipping Faithfulness Evaluation (No eval LLM)\n")
    else:
        print("\nSkipping evaluations in API mode (to save API usage)\n")

    # ---------- CHAT LOOP ----------
    chat_history = deque(maxlen=6)

    print("\n--- Chatbot Ready ---\n")

    while True:
        query = input("Ask: ")

        if query.lower() in ["exit", "quit"]:
            break

        answer, context, docs = run_rag(
            query,
            vectorstore,
            llm,
            reranker,
            bm25,
            chat_history
        )

        print("\n--- Final Answer ---\n")
        print(answer)

        chat_history.append(("user", query))
        chat_history.append(("assistant", answer))


if __name__ == "__main__":
    main()