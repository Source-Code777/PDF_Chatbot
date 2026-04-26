def is_relevant(doc, expected_keywords):
    content = doc.page_content.lower()

    for keyword in expected_keywords:
        if keyword.lower() in content:
            return True

    return False


def precision_at_k(retrieved_docs, expected_keywords, k=3):
    retrieved_docs = retrieved_docs[:k]

    relevant_count = 0

    for doc in retrieved_docs:
        if is_relevant(doc, expected_keywords):
            relevant_count += 1

    return relevant_count / k