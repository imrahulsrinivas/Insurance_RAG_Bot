import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables")


def get_llm(model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
    """
    Initialize and return a ChatOpenAI LLM instance.
    """
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=model_name,
        temperature=temperature,
    )


def build_qa_chain(vector_store, k: int = 5):
    """
    Build a RetrievalQA chain with a custom prompt.
    Uses 'context' and 'question' variables; replies 'I don't know' if out-of-scope.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    llm = get_llm()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an insurance support assistant. Use ONLY the provided document excerpts to answer the question.\n\n
Context:\n{context}\n\n
Question: {question}\n
Answer:"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",
        },
    )