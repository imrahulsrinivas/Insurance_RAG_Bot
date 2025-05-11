from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_split_pdfs(pdf_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Load all PDFs from the specified directory and split them into chunks.
    Returns a list of LangChain Document objects.
    """
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF directory {pdf_dir} not found.")

    docs = []
    for file in pdf_path.glob("*.pdf"):
        loader = PyPDFLoader(str(file))
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)

