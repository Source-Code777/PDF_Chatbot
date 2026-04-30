import os
from langchain_community.vectorstores import Chroma


def get_embedding():
    mode = os.getenv("LLM_MODE", "local")

    if mode == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            encode_kwargs={"normalize_embeddings": True}
        )

    else:
        # Lightweight fallback (no torch)
        from langchain_community.embeddings import FakeEmbeddings

        return FakeEmbeddings(size=768)


def create_vectorstore(chunks, persist_directory):
    embedding = get_embedding()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )

    vectorstore.persist()
    return vectorstore


def load_existing_vectorstore(persist_directory):
    embedding = get_embedding()

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )