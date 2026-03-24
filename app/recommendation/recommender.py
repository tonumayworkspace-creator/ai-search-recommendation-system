from sentence_transformers import SentenceTransformer
import numpy as np

class Recommender:
    def __init__(self, documents):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = documents
        self.texts = [doc["content"] for doc in documents]

        # Generate embeddings
        self.embeddings = self.model.encode(self.texts)

    def recommend(self, doc_id, top_k=3):
        target_index = None

        for i, doc in enumerate(self.documents):
            if doc["id"] == doc_id:
                target_index = i
                break

        if target_index is None:
            return []

        target_embedding = self.embeddings[target_index]

        similarities = np.dot(self.embeddings, target_embedding)

        ranked_indices = similarities.argsort()[::-1]

        recommendations = []
        for idx in ranked_indices:
            if idx != target_index:
                recommendations.append(self.documents[idx])
            if len(recommendations) >= top_k:
                break

        return recommendations