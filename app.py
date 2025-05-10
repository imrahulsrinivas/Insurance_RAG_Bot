import os
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from utils.qa_chain import build_qa_chain

# Must be the first Streamlit command
st.set_page_config(page_title="Insurance PDF Chatbot", layout="wide")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY in .env. Please add it.")
    st.stop()

st.title("📝 Insurance PDF Chatbot")

# Load FAISS index
INDEX_DIR = Path("index.faiss")
if not INDEX_DIR.exists():
    st.error("❌ FAISS index not found. Run `python ingest.py` first.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_vector_store(idx_path: str):
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-ada-002",
    )
    vs = FAISS.load_local(idx_path, embeddings, allow_dangerous_deserialization=True)
    return vs

vector_store = load_vector_store(str(INDEX_DIR))
qa_chain = build_qa_chain(vector_store)

# User input
query = st.text_input("Ask a question about the insurance documents:")
if not query:
    st.info("Enter a question above to get started.")
elif st.button("Ask"):
    with st.spinner("🔍 Retrieving answer…"):
        result = qa_chain({"query": query})
        answer = result.get("result")
        sources = {doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])}
    st.subheader("Answer")
    st.write(answer)
    if sources:
        st.markdown("**Sources:** " + ", ".join(sources))