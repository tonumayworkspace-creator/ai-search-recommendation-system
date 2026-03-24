import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Search System", layout="wide")

st.title("🔍 AI Search & Recommendation System")

# Search Section
st.header("Search")

query = st.text_input("Enter your query")

if st.button("Search"):
    response = requests.get(f"{API_URL}/search", params={"query": query})
    results = response.json().get("results", [])

    st.subheader("Results")
    for doc in results:
        st.write(f"**{doc['title']}**")
        st.write(doc["content"])
        st.write("---")


# Recommendation Section
st.header("Recommendation")

doc_id = st.number_input("Enter Document ID", min_value=1, max_value=10, step=1)

if st.button("Recommend"):
    response = requests.get(f"{API_URL}/recommend", params={"doc_id": doc_id})
    recs = response.json().get("recommendations", [])

    st.subheader("Recommended Documents")
    for doc in recs:
        st.write(f"**{doc['title']}**")
        st.write(doc["content"])
        st.write("---")


# Evaluation Section
st.header("Evaluation Metrics")

if st.button("Run Evaluation"):
    response = requests.get(f"{API_URL}/evaluate")
    metrics = response.json()

    st.write("### Metrics")
    st.json(metrics)