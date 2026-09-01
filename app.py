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


# -------- PAGE CONFIG --------

st.set_page_config(
    page_title="PDF Chatbot",
    layout="wide"
)

st.title("PDF Chatbot")


# -------- MODE SELECTION --------

mode = st.radio(
    "Mode",
    ["Local", "API"]
)


if mode == "API":

    st.info(
        "Enter your Groq API key to use API mode"
    )

    groq_key = st.text_input(
        "Groq API Key",
        type="password"
    )

    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    os.environ["RERANK_MODE"] = "api"
    os.environ["LLM_MODE"] = "api"


else:

    os.environ["RERANK_MODE"] = "local"
    os.environ["LLM_MODE"] = "local"


# -------- FILE UPLOAD --------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# -------- SESSION STATE INIT --------

if "vectorstore" not in st.session_state:

    st.session_state.vectorstore = None

    st.session_state.bm25 = None

    st.session_state.chat_history = []

    st.session_state.current_pdf = None

    st.session_state.llm = None

    st.session_state.reranker = None

    st.session_state.pdf_written = None


# -------- PDF PROCESSING --------

if uploaded_file:

    pdf_name = uploaded_file.name.replace(
        ".pdf",
        ""
    )

    pdf_dir = os.path.join(
        DATA_DIR,
        pdf_name
    )

    db_dir = os.path.join(
        pdf_dir,
        "vectordb"
    )

    os.makedirs(
        pdf_dir,
        exist_ok=True
    )

    pdf_path = os.path.join(
        pdf_dir,
        uploaded_file.name
    )


    # Only write PDF if it is a new file

    if (
        st.session_state.pdf_written
        != uploaded_file.name
    ):

        with open(
            pdf_path,
            "wb"
        ) as f:

            f.write(
                uploaded_file.read()
            )

        st.session_state.pdf_written = (
            uploaded_file.name
        )


    # -------- PROCESS BUTTON --------

    if st.button("Process PDF"):

        with st.spinner(
            "Processing PDF..."
        ):

            # Load PDF

            docs = load_pdf(
                pdf_path
            )


            # Split into chunks

            chunks = split_documents(
                docs
            )


            # Create/load vector database

            if os.path.exists(db_dir):

                vectorstore = (
                    load_existing_vectorstore(
                        db_dir
                    )
                )

            else:

                vectorstore = (
                    create_vectorstore(
                        chunks,
                        db_dir
                    )
                )


            # Create BM25 retriever

            bm25 = BM25Retriever(
                chunks
            )


            # Save everything in session state

            st.session_state.vectorstore = (
                vectorstore
            )

            st.session_state.bm25 = (
                bm25
            )

            st.session_state.chat_history = []

            st.session_state.current_pdf = (
                pdf_name
            )


            # Create LLM

            st.session_state.llm = (
                get_llm()
            )


            # Create reranker

            st.session_state.reranker = (
                Reranker()
            )


        st.success(
            "PDF processed successfully"
        )


# -------- CHAT SECTION --------

if st.session_state.vectorstore:

    st.subheader(
        st.session_state.current_pdf
    )


    # -------- DISPLAY CHAT HISTORY --------

    for role, msg in (
        st.session_state.chat_history
    ):

        with st.chat_message(role):

            st.write(msg)


    # -------- CHAT INPUT --------

    query = st.chat_input(
        "Ask a question"
    )


    if query:

        llm = st.session_state.llm


        # Check LLM

        if llm is None:

            st.error(
                "Please enter your Groq API key "
                "first, then re-process the PDF."
            )

            st.stop()


        # -------- USER MESSAGE --------

        with st.chat_message("user"):

            st.write(query)


        # -------- ASSISTANT RESPONSE --------

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                # Keep last 6 messages
                # = 3 conversation turns

                history_window = deque(
                    st.session_state.chat_history,
                    maxlen=6
                )


                answer, context, docs = run_rag(

                    query,

                    st.session_state.vectorstore,

                    llm,

                    st.session_state.reranker,

                    st.session_state.bm25,

                    history_window
                )


            st.write(answer)


        # -------- SAVE CHAT HISTORY --------

        st.session_state.chat_history.append(
            ("user", query)
        )

        st.session_state.chat_history.append(
            ("assistant", answer)
        )