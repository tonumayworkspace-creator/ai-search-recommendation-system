import streamlit as st
from app.utils.data_loader import load_documents
from app.retrieval.vector_search import VectorSearch
from app.retrieval.bm25_search import BM25Search
from app.retrieval.hybrid_search import HybridSearch
from app.reranking.reranker import ReRanker
from app.recommendation.recommender import Recommender
from app.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Search System",
    page_icon="🔍",
    layout="wide"
)

# ---------------- LOAD SYSTEM ----------------
@st.cache_resource
def load_system():
    documents = load_documents("data/documents.csv")
    vector = VectorSearch(documents)
    bm25 = BM25Search(documents)
    hybrid = HybridSearch(vector, bm25)
    reranker = ReRanker()
    recommender = Recommender(documents)
    return documents, hybrid, reranker, recommender

documents, hybrid_search, reranker, recommender = load_system()

# ---------------- STYLES ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #ffffff;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🔍 AI Search & Recommendation System")
st.markdown("#### ⚡ Hybrid Search • Re-ranking • Recommendations • Evaluation")

st.divider()

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([2, 1])

# ---------------- SEARCH SECTION ----------------
with col1:
    st.subheader("🔎 Search")

    query = st.text_input("Enter your query", placeholder="e.g. machine learning, NLP...")

    if st.button("🚀 Search"):
        if query:
            with st.spinner("Searching..."):
                retrieved = hybrid_search.search(query, top_k=5)
                results = reranker.rerank(query, retrieved, top_k=3)

            st.success("Top Results")

            for doc in results:
                st.markdown(f"""
                <div class="card">
                    <h4>{doc['title']}</h4>
                    <p>{doc['content']}</p>
                </div>
                """, unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with col2:
    st.subheader("🎯 Recommendations")

    doc_id = st.number_input("Document ID", min_value=1, max_value=10, step=1)

    if st.button("Recommend"):
        recs = recommender.recommend(doc_id)

        for doc in recs:
            st.markdown(f"""
            <div class="card">
                <b>{doc['title']}</b><br>
                {doc['content']}
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    st.subheader("📊 Evaluation")

    if st.button("Run Evaluation"):
        query = "machine learning"
        relevant_docs = [1, 2, 9]

        retrieved = hybrid_search.search(query, top_k=5)
        reranked = reranker.rerank(query, retrieved, top_k=5)

        precision = precision_at_k(relevant_docs, reranked, 3)
        recall = recall_at_k(relevant_docs, reranked, 3)
        ndcg = ndcg_at_k(relevant_docs, reranked, 3)

        st.metric("Precision@3", f"{precision:.2f}")
        st.metric("Recall@3", f"{recall:.2f}")
        st.metric("NDCG@3", f"{ndcg:.2f}")