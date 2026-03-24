class HybridSearch:
    def __init__(self, vector_search, bm25_search):
        self.vector_search = vector_search
        self.bm25_search = bm25_search

    def search(self, query, top_k=3):
        vector_results = self.vector_search.search(query, top_k)
        bm25_results = self.bm25_search.search(query, top_k)

        combined = vector_results + bm25_results

        # Remove duplicates
        seen = set()
        unique_results = []

        for doc in combined:
            if doc["id"] not in seen:
                unique_results.append(doc)
                seen.add(doc["id"])

        return unique_results[:top_k]