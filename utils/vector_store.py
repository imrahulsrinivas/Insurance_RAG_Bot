import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Ensure API key is available
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables")

def create_faiss_vector_store(documents, embedding_model: str = "text-embedding-ada-002"):
    """
    Create and return a FAISS vector store from provided Document chunks.
    """
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model=embedding_model,
    )
    return FAISS.from_documents(documents, embeddings)


def load_faiss_vector_store(index_path: str, embeddings):
    """
    Load and return a persisted FAISS vector store with given embeddings.
    """
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)