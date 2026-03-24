import streamlit as st
import requests

# ⚠️ Replace with your deployed backend URL later
API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Search System", layout="wide")

st.title("🔍 AI Search & Recommendation System")

# Search
st.header("Search")
query = st.text_input("Enter your query")

if st.button("Search"):
    try:
        response = requests.get(f"{API_URL}/search", params={"query": query})
        results = response.json().get("results", [])

        for doc in results:
            st.write(f"**{doc['title']}**")
            st.write(doc["content"])
            st.write("---")

    except:
        st.error("Backend not connected ❌")

# Recommendation
st.header("Recommendation")
doc_id = st.number_input("Enter Document ID", min_value=1, max_value=10)

if st.button("Recommend"):
    try:
        response = requests.get(f"{API_URL}/recommend", params={"doc_id": doc_id})
        recs = response.json().get("recommendations", [])

        for doc in recs:
            st.write(f"**{doc['title']}**")
            st.write(doc["content"])
            st.write("---")

    except:
        st.error("Backend not connected ❌")

# Evaluation
st.header("Evaluation")

if st.button("Run Evaluation"):
    try:
        response = requests.get(f"{API_URL}/evaluate")
        st.json(response.json())
    except:
        st.error("Backend not connected ❌")