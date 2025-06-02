from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

import os
import pickle
from typing import List

from dotenv import load_dotenv
import os

# load api
load_dotenv("api.env")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def load_documents(pkl_path: str) -> List[Document]:
    with open(pkl_path, "rb") as f:
        documents = pickle.load(f)
    print(f"✅ Loaded {len(documents)} documents from {pkl_path}")
    return documents


def create_faiss_index(documents: List[Document], output_dir: str = "faiss_index") -> None:
    print("🔍 Generating embeddings and building FAISS index...")
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(output_dir)
    print(f"✅ FAISS index saved to ./{output_dir}")


if __name__ == "__main__":
    docs = load_documents("text/merged_documents.pkl")
    # print(docs[66])
    create_faiss_index(docs, output_dir="faiss_index")
