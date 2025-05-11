# list_external.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import os

emb = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vs = FAISS.load_local("index.faiss", emb, allow_dangerous_deserialization=True)

ext = [
    doc.metadata["source"]
    for doc in vs.docstore._dict.values()
    if doc.metadata["source"].startswith("http")
]
print("External URLs indexed:")
print("\n".join(ext))
