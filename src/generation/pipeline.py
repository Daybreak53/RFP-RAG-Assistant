import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DATASET_PATHS = (
    PROJECT_ROOT / "data" / "eval_dataset_hwp.json",
    PROJECT_ROOT / "data" / "eval_dataset_pdf.json",
)


def _normalize_query(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _load_reference_map():
    reference_map = {}

    for dataset_path in EVAL_DATASET_PATHS:
        if not dataset_path.exists():
            continue

        with dataset_path.open("r", encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            key = _normalize_query(record.get("user_input"))
            reference = record.get("reference")
            if key and reference and key not in reference_map:
                reference_map[key] = reference

    return reference_map


def find_reference_for_query(query):
    reference_map = _load_reference_map()
    normalized_query = _normalize_query(query)

    if normalized_query in reference_map:
        return reference_map[normalized_query]

    compact_query = re.sub(r"\s+", "", normalized_query)
    for eval_query, reference in reference_map.items():
        if re.sub(r"\s+", "", eval_query) == compact_query:
            return reference

    return ""


def rag_pipeline(
        collection_name: str, 
        embed_provider: str, 
        llm_provider: str, 
        query: str, 
        top_k=3, 
        score_threshold=0.2, 
        search_mode="vector", 
        reference=None, 
        gen_params: dict = None
    ):
    from src.retrieval.retriever import retrieve
    from src.generation.gen import generate_answer
    from langfuse import get_client

    langfuse = get_client()

    trace = langfuse.trace(
        name="rag_pipeline",
        input=query,
        metadata={
            "collection_name": collection_name,
            "embed_provider": embed_provider,
            "llm_provider": llm_provider,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "search_mode": search_mode
        }
    )

    retrieval_span = trace.span(
        name="retrieval",
        input={
            "query": query,
            "collection_name": collection_name,
            "embed_provider": embed_provider,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "search_mode": search_mode
        }
    )

    docs = retrieve(collection_name, embed_provider, query, top_k, score_threshold, search_mode)

    retrieval_span.end(
        output=docs
    )

    generation = trace.generation(
        name="answer_generation",
        model=llm_provider,
        input={
            "query": query,
            "retrieved_docs": docs
        }
    )

    answer = generate_answer(
        query,
        docs,
        provider=llm_provider,
        gen_params=gen_params,
    )

    generation.end(
        output=answer
    )

    langfuse.flush()

    return {
        "user_input": query,
        "response": answer,
        "retrieved_context": [d.get("content", "") for d in docs],
        "reference": reference if reference is not None else find_reference_for_query(query)
    }