import streamlit as st
from app.utils.data_loader import load_documents
from app.retrieval.vector_search import VectorSearch
from app.retrieval.bm25_search import BM25Search
from app.retrieval.hybrid_search import HybridSearch
from app.reranking.reranker import ReRanker
from app.recommendation.recommender import Recommender
from app.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k

# Load system
documents = load_documents("data/documents.csv")

vector_search = VectorSearch(documents)
bm25_search = BM25Search(documents)
hybrid_search = HybridSearch(vector_search, bm25_search)
reranker = ReRanker()
recommender = Recommender(documents)

st.title("🔍 AI Search & Recommendation System")

# Search
query = st.text_input("Search")

if st.button("Search"):
    retrieved = hybrid_search.search(query, top_k=5)
    reranked = reranker.rerank(query, retrieved, top_k=3)

    for doc in reranked:
        st.write(f"**{doc['title']}**")
        st.write(doc["content"])
        st.write("---")

# Recommendation
doc_id = st.number_input("Document ID", 1, 10)

if st.button("Recommend"):
    recs = recommender.recommend(doc_id)

    for doc in recs:
        st.write(f"**{doc['title']}**")
        st.write(doc["content"])
        st.write("---")

# Evaluation
if st.button("Run Evaluation"):
    query = "machine learning"
    relevant_docs = [1, 2, 9]

    retrieved = hybrid_search.search(query, top_k=5)
    reranked = reranker.rerank(query, retrieved, top_k=5)

    st.json({
        "precision@3": precision_at_k(relevant_docs, reranked, 3),
        "recall@3": recall_at_k(relevant_docs, reranked, 3),
        "ndcg@3": ndcg_at_k(relevant_docs, reranked, 3)
    })