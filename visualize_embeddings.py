# visualize_embeddings.py
import os
import pandas as pd
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

emb = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vs = FAISS.load_local("index.faiss", emb, allow_dangerous_deserialization=True)

rows = []
for idx in range(vs._faiss_index.ntotal):
    vec = vs._faiss_index.reconstruct(idx)
    doc = vs.docstore._dict[idx]
    rows.append({
        "source": doc.metadata.get("source",""),
        "norm": float(np.linalg.norm(vec)),
        "length": len(doc.page_content),
    })

df = pd.DataFrame(rows)
print(df.head(10))
