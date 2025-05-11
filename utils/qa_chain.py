import os
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.tracers import LangChainTracer
from langchain.chat_models import ChatOpenAI


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables")

# Tracers for observability
# Instantiate tracers (no args required for LangChainTracer)
llm_tracer = LangChainTracer()
chain_tracer = LangChainTracer()

def get_llm(model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
    """
    Initialize and return a ChatOpenAI LLM instance with LangSmith tracing.
    """
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=model_name,
        temperature=temperature,
        callbacks=[llm_tracer],  # trace LLM calls
    )


def build_qa_chain(vector_store, k: int = 5):
    """
    Build and return a RetrievalQA chain with chain-level LangSmith tracing.
    Uses 'context' and 'question' variables; replies 'I don't know' if out of scope.
    """
    # Retriever and LLM
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    llm = get_llm()

    # Define prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an insurance support assistant. Use ONLY the provided document excerpts to answer the question.\n\n
Context:\n{context}\n\n
Question: {question}\n
Answer:"""
    )

    # Build chain with tracing
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        callbacks=[chain_tracer],  # trace chain steps
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",
        },
    )