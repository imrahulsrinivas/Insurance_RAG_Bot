from dotenv import load_dotenv
import os
from pathlib import Path

from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Configuration
PDF_DIR = Path("data/insurance_pdfs")  # Ensure this matches your folder name
INDEX_DIR = Path("index.faiss")
EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables")


def load_pdfs(pdf_dir: Path):
    """
    Load all PDF files from the directory into LangChain Document objects.
    """
    docs = []
    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"Loading {pdf_file}...")
        loader = PyPDFLoader(str(pdf_file))
        docs.extend(loader.load())
    return docs


def split_documents(docs, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split large documents into smaller, overlapping chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def build_faiss_index(docs):
    """
    Build and return a FAISS vector store from document chunks.
    """
    print("Initializing embeddings...")
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL,
    )
    print("Building FAISS index...")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def main():
    # Load and split documents
    raw_docs = load_pdfs(PDF_DIR)
    print(f"Loaded {len(raw_docs)} pages.")
    split_docs = split_documents(raw_docs)
    print(f"Split into {len(split_docs)} chunks.")

    # Extract & fetch external URLs
    from utils.external_loader import load_external_documents_from_pdfs
    external_docs = load_external_documents_from_pdfs(split_docs)
    print(f"Found {len(external_docs)} external documents/URLs.")

    # Load & split any external PDFs, or keep HTML/text docs
    from langchain_community.document_loaders import PyPDFLoader
    new_chunks = []
    for doc in external_docs:
        src = doc.metadata.get("source", "")
        if src.lower().endswith(".pdf"):
            loader = PyPDFLoader(src)
            new_chunks.extend(loader.load_and_split())
        else:
            new_chunks.append(doc)

    all_docs = split_docs + new_chunks
    print(f"Total chunks (incl. external): {len(all_docs)}")

    # Build & save the combined index
    vector_store = build_faiss_index(all_docs)



    # Build and save index
    vector_store = build_faiss_index(split_docs)
    print(f"Saving index to {INDEX_DIR}...")
    vector_store.save_local(str(INDEX_DIR))
    print("Index saved!")


if __name__ == "__main__":
    main()