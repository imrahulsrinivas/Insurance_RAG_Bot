Insurance RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for insurance customer support documentation. It ingests PDF files (and optionally external links found within them), builds a FAISS vector index of content chunks, and provides a Streamlit-based chat interface that answers questions strictly from the indexed sources—replying “I don’t know” for out-of-scope queries.

📂 Repository Structure

Insurance_RAG_Bot/
├── .gitignore               # ignore .env, .venv, index.faiss
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── .venv/                   # Python virtual environment (local)
├── app.py                   # Streamlit UI
├── ingest.py                # Ingest PDFs & external links, build index
├── list_external.py         # CLI: list external URLs in the index
├── test_retrieval.py        # CLI: interactive QA testing
├── visualize_embeddings.py  # CLI: inspect embedding norms & lengths
├── data/                    # Folder containing your PDF sources
│   └── insurance_pdfs/
├── index.faiss/             # Persisted FAISS index (ignored)
└── utils/
    ├── loader.py            # PDF loading & splitting
    ├── vector_store.py      # FAISS index creation & loading
    ├── qa_chain.py          # RetrievalQA chain + LangSmith tracing
    └── external_loader.py   # Extract & fetch external URLs

🚀 Local Setup & Run

Clone this repo:

git clone https://github.com/imrahulsrinivas/Insurance_RAG_Bot.git
cd Insurance_RAG_Bot

Create & activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install --upgrade pip
pip install -r requirements.txt

Configure your API keys in a .env at project root:

OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=ls-...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=Insurance_RAG_Bot

Ingest your source documents:

python ingest.py

Loads all PDFs under data/insurance_pdfs/

Extracts external URLs inside PDF text, fetches and indexes them

Splits text into chunks, builds FAISS index at index.faiss/

Launch the chatbot UI:

streamlit run app.py

Open the printed Local URL (e.g. http://localhost:8501) in your browser.

🛠️ Helper Scripts

list_external.py: List all external URLs indexed.

python list_external.py

test_retrieval.py: Interactive CLI to ask questions and view sources.

python test_retrieval.py

visualize_embeddings.py: Dump a table of embedding norms and chunk lengths.

python visualize_embeddings.py

🌐 Deployment to Streamlit Community Cloud

Push your clean main branch to GitHub (ensure .env is in .gitignore).

On https://share.streamlit.io, click Create app.

Repository: imrahulsrinivas/Insurance_RAG_Bot

Branch: main

App file path: app.py

In the Secrets section of the app settings, add:

OPENAI_API_KEY = "sk-..."
LANGSMITH_API_KEY = "ls-..."
LANGSMITH_TRACING = "true"
LANGSMITH_PROJECT = "Insurance_RAG_Bot"

Click Deploy. Your live app URL will be displayed once ready.

📈 Monitoring & Tracing with LangSmith

The QA chain and LLM calls are instrumented with LangChainTracer callbacks.

Ensure LANGSMITH_TRACING=true in your environment.

Visit https://app.langsmith.com and select project Insurance_RAG_Bot to view runs:

Runs: each RetrievalQA invocation with inputs, retrieved chunks, and outputs.

LLM Calls: raw prompts, completions, and token usage.
