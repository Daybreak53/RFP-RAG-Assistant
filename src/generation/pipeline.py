from src.retrieval.retriever import retrieve
from src.generation.gen import generate_answer

def rag_pipeline(collection_name: str, embed_provider: str, llm_provider: str, query: str):

    docs = retrieve(collection_name, embed_provider, query)

    answer = generate_answer(
        query,
        docs,
        provider=llm_provider
    )

    return {
        "user_input": query,
        "response": answer,
        "retrieved_context": [d.get("content", "") for d in docs],
        "reference": "AWS 기반 클라우드 전환 사업이다. Kubernetes 운영 경험이 필요하다."
    }