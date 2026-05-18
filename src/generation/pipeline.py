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
    run_eval=False,
    eval_model_name="gpt-4o-mini",
    eval_is_local=False
    ):

    from src.retrieval.retriever import retrieve
    from src.generation.gen import generate_answer
    from langfuse import get_client


    langfuse = get_client()

    with langfuse.start_as_current_observation(
        name="rag_pipeline",
        as_type="span",
        input=query,
        metadata={
            "collection_name": collection_name,
            "embed_provider": embed_provider,
            "llm_provider": llm_provider,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "search_mode": search_mode
        }
    ) as pipeline_span:

        with langfuse.start_as_current_observation(
            name="retrieval",
            as_type="span",
            input={
                "query": query,
                "collection_name": collection_name,
                "embed_provider": embed_provider,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "search_mode": search_mode
            }
        ) as retrieval_span:

            docs = retrieve(collection_name, embed_provider, query, top_k, score_threshold, search_mode)

            retrieval_span.update(output=docs)

        with langfuse.start_as_current_observation(
            name="answer_generation",
            as_type="generation",
            model=llm_provider,
            input={
                "query": query,
                "retrieved_docs": docs
            }
        ) as generation:

            answer = generate_answer(
                query,
                docs,
                provider=llm_provider
            )

            generation.update(output=answer)

        result = {
            "user_input": query,
            "response": answer,
            "retrieved_context": [d.get("content", "") for d in docs],
            "reference": reference if reference is not None else find_reference_for_query(query)
        }

        if run_eval:
            from src.evaluation.evaluate import evaluate

            with langfuse.start_as_current_observation(
                name="ragas_evaluation",
                as_type="span",
                input=result
            ) as eval_span:

                eval_output = evaluate(
                    evaluation_data=[result],
                    model_name=eval_model_name,
                    is_local=eval_is_local
                )

                eval_result = eval_output.to_dict(orient="records")

                eval_span.update(output=eval_result)

                result["evaluation"] = eval_result

        pipeline_span.update(output=result)

    langfuse.flush()

    return result
