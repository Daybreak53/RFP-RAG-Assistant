from rag.retriever import retrieve
from rag.gen import generate_answer
from rag.config import LLM_PROVIDER

def rag_pipeline(query):

    docs = retrieve(query)

    answer = generate_answer(
        query,
        docs,
        provider=LLM_PROVIDER
    )

    return answer