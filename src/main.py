from loader import load_pdf
from splitter import split_documents
from vectorstore import create_vectorstore
from llm import get_llm, generate_answer
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
DB_DIR="db"
def load_existing_vectorstore_data():
    embedding=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore=Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding
    )
    return vectorstore

def generate_query_variations(llm, query):

    prompt = f"""
You are a helpful assistant that rewrites user questions for better document retrieval.

Generate 3 different rephrasings of the following question.
Return ONLY the questions, one per line.

Question: {query}
"""
    response = llm(prompt)
    text = response[0]["generated_text"]
    variations = text.replace(prompt, "").strip().split("\n")
    variations.append(query)
    return [q.strip() for q in variations if q.strip()]

def rerank_documents(query, docs, embedding_model, top_k=3):
    query_embedding = embedding_model.embed_query(query)

    scored_docs = []

    for doc in docs:
        doc_embedding = embedding_model.embed_query(doc.page_content)
        score = cosine_similarity(
            [query_embedding],
            [doc_embedding]
        )[0][0]

        scored_docs.append((score, doc))

    # sort by score (descending)
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # pick top_k
    top_docs = [doc for _, doc in scored_docs[:top_k]]

    return top_docs


if __name__ == "__main__":
    path=r"C:\Users\aasim\OneDrive\Desktop\Notes\NLP\cs224n_winter2023_lecture1_notes_draft.pdf"

    if os.path.exists(DB_DIR):
        print("Loading existing vectorstore database")
        vectorstore=load_existing_vectorstore_data()
    else:
        print("Creating new vectorstore database")
        vectorstore=load_pdf(path)
        chunks=split_documents(vectorstore)
        vectorstore=create_vectorstore(chunks)


    llm = get_llm()
    chat_history=[]

    while True:
        query=input("\nAsk: ")
        if query.lower() in ["exit","quit"]:
            break

        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        queries=generate_query_variations(llm,query)
        all_docs=[]
        for q in queries:
            docs=retriever.invoke(q)
            all_docs.extend(docs)
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        results = rerank_documents(query, unique_docs, embedding_model, top_k=3)
        context = "\n\n---\n\n".join([doc.page_content for doc in results])
        answer = generate_answer(llm, query, context, chat_history)
        print("\n---Final Answer---\n")
        print(answer)
        chat_history.append(("user",query))
        chat_history.append(("Assistant",answer))
        chat_history=chat_history[-6:]

