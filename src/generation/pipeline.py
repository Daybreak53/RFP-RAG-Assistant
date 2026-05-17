from src.retrieval.retriever import retrieve
from src.generation.gen import generate_answer
from src.core.config import LLM_PROVIDER
from langfuse import get_client


def rag_pipeline(query):

    langfuse = get_client()

    trace = langfuse.trace(
        name="rag_pipeline",
        input=query,
        metadata={
            "llm_provider": LLM_PROVIDER
        }
    )

    retrieval_span = trace.span(
        name="retrieval",
        input=query
    )

    docs = retrieve(query)

    retrieval_span.end(
        output=docs
    )

    generation = trace.generation(
        name="answer_generation",
        model=LLM_PROVIDER,
        input={
            "query": query,
            "retrieved_docs": docs
        }
    )

    answer = generate_answer(
        query,
        docs,
        provider=LLM_PROVIDER
    )

    generation.end(
        output=answer
    )

    langfuse.flush()

    return {
        "user_input": query,
        "response": answer,
        "retrieved_context": [d.get("content", "") for d in docs],
        "reference": "AWS 기반 클라우드 전환 사업이다. Kubernetes 운영 경험이 필요하다."
    }