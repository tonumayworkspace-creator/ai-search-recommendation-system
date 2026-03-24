import pandas as pd

def load_documents(file_path: str):
    df = pd.read_csv(file_path)
    documents = []

    for _, row in df.iterrows():
        doc = {
            "id": row["id"],
            "title": row["title"],
            "content": row["content"],
            "category": row["category"]
        }
        documents.append(doc)

    return documents