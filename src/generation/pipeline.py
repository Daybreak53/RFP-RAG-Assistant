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

    return {
        "user_input": query,
        "response": answer,
        "retrieved_context": [d.get("content", "") for d in docs], 
        "reference": "AWS 기반 클라우드 전환 사업이다. Kubernetes 운영 경험이 필요하다." 
    }