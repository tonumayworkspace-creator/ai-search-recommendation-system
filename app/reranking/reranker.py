from sentence_transformers import CrossEncoder

class ReRanker:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank(self, query, documents, top_k=3):
        pairs = [(query, doc["content"]) for doc in documents]

        scores = self.model.predict(pairs)

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        reranked = [doc for doc, score in scored_docs[:top_k]]

        return reranked