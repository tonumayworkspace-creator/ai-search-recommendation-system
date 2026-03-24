from rank_bm25 import BM25Okapi

class BM25Search:
    def __init__(self, documents):
        self.documents = documents
        self.corpus = [doc["content"].split() for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query, top_k=3):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices[:top_k]:
            results.append(self.documents[idx])

        return results