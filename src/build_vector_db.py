# src/build_vector_db.py

import os

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

DATASET_DIR = "./dataset"
CHROMA_DIR = "./vector_db"
CHROMA_COLLECTION = "product_catalog"


def make_product_embedding_text(product_name, aisle, department):
    return f"{product_name}, found in the {aisle} aisle of the {department} department."


def load_and_prepare_product_catalog(dataset_dir=DATASET_DIR):
    products = pd.read_csv(os.path.join(dataset_dir, "products.csv"))
    aisles = pd.read_csv(os.path.join(dataset_dir, "aisles.csv"))
    departments = pd.read_csv(os.path.join(dataset_dir, "departments.csv"))

    # Filter: keep only rows where aisle_id and department_id are both numeric
    is_aisle_numeric = products["aisle_id"].astype(str).str.strip().str.isnumeric()
    is_dept_numeric = products["department_id"].astype(str).str.strip().str.isnumeric()
    products = products[is_aisle_numeric & is_dept_numeric].copy()

    # Cast merge keys to int
    products["aisle_id"] = products["aisle_id"].astype(int)
    products["department_id"] = products["department_id"].astype(int)

    products = products[["product_id", "product_name", "aisle_id", "department_id"]]

    merged = products.merge(aisles, on="aisle_id", how="left").merge(
        departments, on="department_id", how="left"
    )
    merged["embedding_text"] = merged.apply(
        lambda row: make_product_embedding_text(
            row["product_name"], row["aisle"], row["department"]
        ),
        axis=1,
    )
    return merged[
        ["product_id", "product_name", "aisle", "department", "embedding_text"]
    ]


def make_langchain_documents(df):
    docs = [
        Document(
            page_content=row["embedding_text"],
            metadata={
                "product_id": int(row["product_id"]),
                "product_name": row["product_name"],
                "aisle": row["aisle"],
                "department": row["department"],
            },
        )
        for _, row in df.iterrows()
    ]
    return docs


from tqdm import tqdm


def build_and_persist_chroma(docs, persist_directory=CHROMA_DIR, batch_size=256):

    print(f"Embedding {len(docs)} documents in batches of {batch_size}...")

    embeddings = HuggingFaceEmbeddings(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    # Chunk docs for progress reporting
    for i in tqdm(range(0, len(docs), batch_size)):
        batch = docs[i : i + batch_size]
        vector_store.add_documents(documents=batch)
    print(f"Chroma DB built and persisted at: {persist_directory}")


if __name__ == "__main__":
    df = load_and_prepare_product_catalog()
    docs = make_langchain_documents(df)
    build_and_persist_chroma(docs)
    print("Vector DB built and ready.")
