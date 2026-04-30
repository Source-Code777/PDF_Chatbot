import os
import streamlit as st
from collections import deque

from src.loader import load_pdf
from src.splitter import split_documents
from src.vectorstore import create_vectorstore, load_existing_vectorstore
from src.retrieval.bm25 import BM25Retriever
from src.core.rag_pipeline import run_rag
from src.llm import get_llm
from src.reranker import Reranker

DATA_DIR = "data"

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("PDF Chatbot")

# -------- MODE SELECTION --------
mode = st.radio("Mode", ["Local", "API"])

if mode == "API":
    st.info("Enter your Groq API key to use API mode")

    groq_key = st.text_input("Groq API Key", type="password")

    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    os.environ["RERANK_MODE"] = "api"
    os.environ["LLM_MODE"] = "api"
else:
    os.environ["RERANK_MODE"] = "local"
    os.environ["LLM_MODE"] = "local"

# -------- FILE UPLOAD --------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# -------- SESSION STATE INIT --------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.bm25 = None
    st.session_state.chat_history = deque(maxlen=6)
    st.session_state.current_pdf = None

# IMPORTANT: initialize processing flag separately
if "processing" not in st.session_state:
    st.session_state.processing = False

# -------- PDF PROCESSING --------
if uploaded_file:
    pdf_name = uploaded_file.name.replace(".pdf", "")
    pdf_dir = os.path.join(DATA_DIR, pdf_name)
    db_dir = os.path.join(pdf_dir, "vectordb")

    os.makedirs(pdf_dir, exist_ok=True)

    pdf_path = os.path.join(pdf_dir, uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            docs = load_pdf(pdf_path)
            chunks = split_documents(docs)

            # Avoid recreating vectorstore repeatedly
            if os.path.exists(db_dir):
                vectorstore = load_existing_vectorstore(db_dir)
            else:
                vectorstore = create_vectorstore(chunks, db_dir)

            bm25 = BM25Retriever(chunks)

            st.session_state.vectorstore = vectorstore
            st.session_state.bm25 = bm25
            st.session_state.chat_history.clear()
            st.session_state.current_pdf = pdf_name

        st.success("PDF processed successfully")

# -------- CHAT SECTION --------
if st.session_state.vectorstore:

    st.subheader(st.session_state.current_pdf)

    # Show chat history
    for role, msg in st.session_state.chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.write(msg)

    # Chat input
    query = st.chat_input("Ask a question")

    if query and not st.session_state.processing:
        st.session_state.processing = True

        try:
            llm = get_llm()

            if llm is None:
                st.error("Please enter your Groq API key")
                st.session_state.processing = False
                st.stop()

            reranker = Reranker()

            with st.spinner("Thinking..."):
                answer, context, docs = run_rag(
                    query,
                    st.session_state.vectorstore,
                    llm,
                    reranker,
                    st.session_state.bm25,
                    st.session_state.chat_history
                )

            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("assistant", answer))

        finally:
            # ALWAYS reset processing (even if error happens)
            st.session_state.processing = False