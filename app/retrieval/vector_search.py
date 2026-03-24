from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorSearch:
    def __init__(self, documents):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = documents
        self.texts = [doc["content"] for doc in documents]

        # Generate embeddings
        self.embeddings = self.model.encode(self.texts)

        # Create FAISS index
        self.dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(self.embeddings))

    def search(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)

        results = []
        for idx in indices[0]:
            results.append(self.documents[idx])

        return results