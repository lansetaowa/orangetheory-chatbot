from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from fpdf import FPDF
import os
from collections import defaultdict
from urllib.parse import urlparse

from langchain_community.document_loaders import WikipediaLoader

def load_wikipedia_documents(query: str = "Orangetheory Fitness", lang: str = "en"):
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=2)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} Wikipedia documents.")
    return docs

# import os
# os.environ["USER_AGENT"] = "OrangetheoryBot/0.1 (+https://github.com/lansetaowa/orangetheory-chatbot)"

# Define the Orangetheory URLs to crawl
ORANGETHEORY_URLS = [
    "https://www.orangetheory.com/en-us/workout",
    "https://www.orangetheory.com/en-us/memberships",
    "https://www.orangetheory.com/en-us/our-commitment",
    "https://www.orangetheory.com/en-us/faq",
    "https://www.orangetheory.com/en-us/app-faq",
    "https://www.orangetheory.com/en-us/product-information",
    "https://www.orangetheory.com/en-us/press",
    "https://www.orangetheory.com/en-us/articles",
    "https://www.orangetheory.com/en-us/promotion-terms",
]

def fetch_and_split_documents(urls: List[str], chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Load webpages from the given list of URLs and split them into smaller chunks.

    Args:
        urls (List[str]): List of URLs to load.
        chunk_size (int): Maximum size of each text chunk.
        chunk_overlap (int): Overlap size between chunks.

    Returns:
        List[Document]: List of split document chunks.
    """
    loader = WebBaseLoader(urls)
    raw_documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    split_docs = splitter.split_documents(raw_documents)
    return split_docs

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_font("DejaVu", "", "font/DejaVuSans.ttf", uni=True)
        self.set_font("DejaVu", size=12)

def save_documents_grouped_by_source(documents, output_dir="text"):
    os.makedirs(output_dir, exist_ok=True)
    grouped = defaultdict(list)

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        grouped[source].append(doc.page_content)

    for source_url, chunks in grouped.items():
        parsed_url = urlparse(source_url)
        name = parsed_url.path.strip("/").replace("/", "_") or "homepage"
        pdf_path = os.path.join(output_dir, f"{name}.pdf")

        pdf = PDF()

        for i, content in enumerate(chunks):
            pdf.add_page()
            pdf.multi_cell(0, 10, f"--- Chunk {i+1} ---\n\n{content}")

        pdf.output(pdf_path)
        print(f"✅ Saved {len(chunks)} chunks to {pdf_path}")


if __name__ == "__main__":
    import pickle

    web_documents = fetch_and_split_documents(ORANGETHEORY_URLS)
    print(f"Loaded and split {len(web_documents)} chunks from Orangetheory website.")
    #
    # # Optionally save or inspect chunks here
    # save_documents_grouped_by_source(documents, output_dir="text")
    #
    # for i, doc in enumerate(documents[:5]):
    #     print(f"--- Chunk {i+1} ---")
    #     print(doc.page_content[:500])
    #     print("\n")

    wiki_docs = load_wikipedia_documents()
    # save_documents_grouped_by_source(wiki_docs, output_dir='text')

    all_docs = web_documents + wiki_docs

    with open("text/merged_documents.pkl", "wb") as f:
        pickle.dump(all_docs, f)
