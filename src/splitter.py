from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    total_length = sum(len(doc.page_content) for doc in documents)
    if total_length < 20000:
        chunk_size = 300
        chunk_overlap = 50
    elif total_length < 100000:
        chunk_size = 500
        chunk_overlap = 50
    else:
        chunk_size = 800
        chunk_overlap = 100
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks=splitter.split_documents(documents)
    return chunks