# Insurance RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for customer support, built with LangChain and Streamlit. It answers user questions based only on provided PDF documents (and optional external links), falling back to "I don't know" for out-of-scope queries.

---

## 📂 Repository Structure

```
Alltius_Assignment_Rahul/
├── ingest.py                # Ingests PDFs & external links, builds FAISS index
├── app.py                   # Streamlit UI for the chatbot
├── utils/
│   ├── loader.py            # PDF loading & splitting helpers
│   ├── vector_store.py      # FAISS creation & loading helpers
│   ├── qa_chain.py          # RetrievalQA chain with LangSmith tracing
│   └── external_loader.py   # URL extraction & fetching logic
├── data/
│   └── insurance_pdfs/      # Your source PDF files
├── .env                     # API keys (ignored via .gitignore)
├── requirements.txt         # Python dependencies
├── index.faiss/             # Persisted FAISS index (ignored)
└── README.md                # This documentation
```

---

## 🚀 Installation & Local Run

1. **Clone** the repo:

   ```bash
   git clone https://github.com/imrahulsrinivas/Insurance_RAG_Bot.git
   cd Insurance_RAG_Bot
   ```

2. **Create & activate** a Python virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install** dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure** environment variables in `.env`:

   ```ini
   OPENAI_API_KEY=sk-...
   LANGSMITH_API_KEY=ls-...
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=Insurance_RAG_Bot
   ```

5. **Build** the vector index (PDFs + external links):

   ```bash
   python ingest.py
   ```

6. **Launch** the Streamlit app:

   ```bash
   streamlit run app.py
   ```

Visit `http://localhost:8501` in your browser.

---

## 🌐 Deployment to Streamlit Community Cloud

1. **Push** your cleaned repo to GitHub (ensure `.env` is in `.gitignore`).
2. On [https://streamlit.io/cloud](https://streamlit.io/cloud), click **New app** → select your GitHub repo & branch.
3. In the **Secrets** tab on Streamlit, add your keys:

   * `OPENAI_API_KEY`
   * `LANGSMITH_API_KEY`
   * `LANGSMITH_TRACING=true`
   * `LANGSMITH_PROJECT=Insurance_RAG_Bot`
4. Click **Deploy**. Your chatbot will be live at a `streamlit.app` URL.

---

## 🔧 Test Cases

Use the following to validate functionality:

| Question                                    | Expected Behavior                          |
| ------------------------------------------- | ------------------------------------------ |
| "What is the deductible for the Gold plan?" | Answers based on PDF content               |
| "How do I file a claim?"                    | Answers or 'I don't know' if not in source |
| "What does HOMEPAGE link refer to?"         | External URL content used to answer        |
| "Who is the current CEO of OpenAI?"         | "I don't know" (out-of-scope query)        |

### Automated CLI Test

Create `test_retrieval.py`:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from utils.qa_chain import build_qa_chain
import os

# Load index
o = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vs = FAISS.load_local("index.faiss", o, allow_dangerous_deserialization=True)
qa = build_qa_chain(vs)

# Sample prompts
tests = [
    ("What is the deductible for the Gold plan?", None),
    ("Who is the CEO of OpenAI?", "I don't know"),
]

for q, expected in tests:
    res = qa({"query": q})
    print(f"Q: {q}")
    print(f"A: {res['result']}")
    if expected:
        print("PASS" if expected in res['result'] else "FAIL")
    print("---")
```

Run:

```bash
python test_retrieval.py
```

---

## 📈 Monitoring & Tracing with LangSmith

* Ensure `.env` has `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT`.
* PylChain `LangChainTracer` in `utils/qa_chain.py` will send both LLM calls and chain steps to LangSmith.
* Interact with the Streamlit app; then view detailed run traces at [https://app.langsmith.com](https://app.langsmith.com) under your project.

---

*End of documentation.*
