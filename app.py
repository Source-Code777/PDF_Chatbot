import os
import streamlit as st
from collections import deque

from src.loader import load_pdf
from src.splitter import split_documents
from src.vectorstore import create_vectorstore
from src.retrieval.bm25 import BM25Retriever
from src.core.rag_pipeline import run_rag
from src.llm import get_llm
from src.reranker import CrossEncoderReranker

DATA_DIR = "data"

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 PDF Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.bm25 = None
    st.session_state.chat_history = deque(maxlen=6)
    st.session_state.current_pdf = None

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

            vectorstore = create_vectorstore(chunks, db_dir)
            bm25 = BM25Retriever(chunks)

            st.session_state.vectorstore = vectorstore
            st.session_state.bm25 = bm25
            st.session_state.chat_history.clear()
            st.session_state.current_pdf = pdf_name

        st.success("PDF processed successfully!")

if st.session_state.vectorstore:

    st.divider()
    st.subheader(f"Chat with: {st.session_state.current_pdf}")

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")

    query = st.text_input("Ask a question")

    if query:
        llm = get_llm()
        reranker = CrossEncoderReranker()

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

        st.rerun()