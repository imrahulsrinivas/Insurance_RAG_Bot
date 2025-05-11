# test_retrieval.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from utils.qa_chain import build_qa_chain
import os

emb = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vs = FAISS.load_local("index.faiss", emb, allow_dangerous_deserialization=True)
qa = build_qa_chain(vs)

print("Type a question (or 'exit'):")
while True:
    q = input("➜ ")
    if q.lower()=="exit": break
    res = qa({"query": q})
    print("Answer:", res["result"])
    print("Sources:", [d.metadata["source"] for d in res["source_documents"]])
    print("-"*40)
