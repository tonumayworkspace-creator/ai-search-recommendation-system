import numpy as np

def precision_at_k(relevant_docs, retrieved_docs, k):
    retrieved_k = retrieved_docs[:k]
    relevant_set = set(relevant_docs)

    hits = sum([1 for doc in retrieved_k if doc["id"] in relevant_set])

    return hits / k


def recall_at_k(relevant_docs, retrieved_docs, k):
    retrieved_k = retrieved_docs[:k]
    relevant_set = set(relevant_docs)

    hits = sum([1 for doc in retrieved_k if doc["id"] in relevant_set])

    return hits / len(relevant_docs) if relevant_docs else 0


def ndcg_at_k(relevant_docs, retrieved_docs, k):
    dcg = 0.0
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc["id"] in relevant_docs:
            dcg += 1 / np.log2(i + 2)

    ideal_dcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_docs), k))])

    return dcg / ideal_dcg if ideal_dcg > 0 else 0