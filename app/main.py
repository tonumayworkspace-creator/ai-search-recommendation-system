from fastapi import FastAPI
from app.utils.data_loader import load_documents
from app.retrieval.vector_search import VectorSearch
from app.retrieval.bm25_search import BM25Search
from app.retrieval.hybrid_search import HybridSearch
from app.reranking.reranker import ReRanker
from app.recommendation.recommender import Recommender
from app.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k

app = FastAPI(
    title="AI Search & Recommendation System",
    description="LLM + Search + Recommendation Engine",
    version="1.0"
)

# Load documents
documents = load_documents("data/documents.csv")

# Initialize components
vector_search = VectorSearch(documents)
bm25_search = BM25Search(documents)
hybrid_search = HybridSearch(vector_search, bm25_search)
reranker = ReRanker()
recommender = Recommender(documents)


@app.get("/")
def root():
    return {"message": "AI Search Recommendation System is Running 🚀"}


@app.get("/health")
def health_check():
    return {"status": "OK"}


@app.get("/documents")
def get_documents():
    return {"total_documents": len(documents), "data": documents}


@app.get("/search")
def search(query: str):
    retrieved_docs = hybrid_search.search(query, top_k=5)
    reranked_docs = reranker.rerank(query, retrieved_docs, top_k=3)

    return {
        "query": query,
        "results": reranked_docs
    }


@app.get("/recommend")
def recommend(doc_id: int):
    results = recommender.recommend(doc_id)

    return {
        "doc_id": doc_id,
        "recommendations": results
    }


@app.get("/evaluate")
def evaluate():
    # Simulated ground truth
    query = "machine learning"
    relevant_docs = [1, 2, 9]  # known relevant IDs

    retrieved_docs = hybrid_search.search(query, top_k=5)
    reranked_docs = reranker.rerank(query, retrieved_docs, top_k=5)

    precision = precision_at_k(relevant_docs, reranked_docs, k=3)
    recall = recall_at_k(relevant_docs, reranked_docs, k=3)
    ndcg = ndcg_at_k(relevant_docs, reranked_docs, k=3)

    return {
        "query": query,
        "precision@3": precision,
        "recall@3": recall,
        "ndcg@3": ndcg
    }