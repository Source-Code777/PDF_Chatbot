from loader import load_pdf
from splitter import split_documents
from vectorstore import create_vectorstore
from llm import get_llm, generate_answer

if __name__ == "__main__":
    path=r"C:\Users\aasim\OneDrive\Desktop\Notes\NLP\cs224n_winter2023_lecture1_notes_draft.pdf"
    docs = load_pdf(path)
    chunks = split_documents(docs)

    vectorstore = create_vectorstore(chunks)

    query = "What is Word2Vec?"

    results = vectorstore.similarity_search(query, k=3)


    context = "\n\n".join([doc.page_content for doc in results])


    llm = get_llm()

    answer = generate_answer(llm, query, context)

    print("\n--- FINAL ANSWER ---\n")
    print(answer)