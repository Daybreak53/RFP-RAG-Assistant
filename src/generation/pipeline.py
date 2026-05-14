from src.retrieval.retriever import retrieve
from src.generation.gen import generate_answer
from src.core.config import LLM_PROVIDER

def rag_pipeline(query):

    docs = retrieve(query)

    answer = generate_answer(
        query,
        docs,
        provider=LLM_PROVIDER
    )

    return answer