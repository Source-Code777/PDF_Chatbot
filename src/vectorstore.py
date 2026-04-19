from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def create_vectorstore(chunks):
    embedding=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embedding
    )
    return vectorstore