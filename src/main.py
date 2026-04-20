from loader import load_pdf
from splitter import split_documents
from vectorstore import create_vectorstore
from llm import get_llm, generate_answer

if __name__ == "__main__":
    path=r"C:\Users\aasim\OneDrive\Desktop\Notes\NLP\cs224n_winter2023_lecture1_notes_draft.pdf"
    docs = load_pdf(path)
    chunks = split_documents(docs)
    vectorstore = create_vectorstore(chunks)
    llm = get_llm()
    chat_history=[]

    while True:
        query=input("\nAsk: ")
        if query.lower() in ["exit","quit"]:
            break
        results=vectorstore.similarity_search(query,k=3)
        context="\n\n".join([doc.page_content for doc in results])
        answer=generate_answer(llm,context,query,chat_history)
        print("\n---Final Answer---\n")
        print(answer)
        chat_history.append(("user",query))
        chat_history.append(("Assistant",answer))
        chat_history=chat_history[-6:]

