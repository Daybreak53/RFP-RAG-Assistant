import json
import re
from pathlib import Path
from typing import Optional
from src.retrieval.retriever import retrieve
from src.retrieval.filter_extractor import resolve_filter
from src.generation.gen import generate_answer
from langfuse import get_client
from src.evaluation.evaluate import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVAL_DATASET_PATHS = (
    PROJECT_ROOT / "data" / "eval_dataset.json",
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


SOURCE_REVIEW_STATUSES = {"missing_citation", "bad_source_location", "unclear"}


def _verify_answer_sources(answer, docs, fallback_top_k=5):
    from source_check import flatten_chunk, verify_answer

    chunks = [flatten_chunk(doc) for doc in docs]
    if not chunks:
        return {
            "overall_status": "needs_review",
            "source_mode": "no_retrieved_docs",
            "citations": {"chunk_ids": [], "file_names": [], "pages": []},
            "claim_count": 0,
            "checks": [
                {
                    "status": "missing_citation",
                    "reason": "No retrieved documents were available for source verification.",
                    "claim": "",
                }
            ],
        }

    return verify_answer(answer, chunks, fallback_top_k)


def _source_verification_issues(verification):
    return [
        check
        for check in verification.get("checks", [])
        if check.get("status") in SOURCE_REVIEW_STATUSES
    ]


def _build_source_warning(verification):
    if verification.get("overall_status") == "not_found_response":
        return ""

    issues = _source_verification_issues(verification)
    if not issues:
        return ""

    status_counts = {}
    for issue in issues:
        status = issue.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    issue_summary = ", ".join(
        f"{status} {count}건"
        for status, count in sorted(status_counts.items())
    )
    lines = [
        "[출처 검증 경고]",
        f"- 검증 상태: {verification.get('overall_status')}",
        f"- 문제 유형: {issue_summary}",
        "- 일부 답변은 인용 출처와 충분히 일치하지 않아 근거 부족으로 검토가 필요합니다.",
    ]

    for issue in issues[:3]:
        claim = str(issue.get("claim") or "").strip()
        if len(claim) > 100:
            claim = claim[:100].rstrip() + "..."
        if claim:
            lines.append(f"- 확인 필요: {issue.get('status')} | {claim}")

    return "\n".join(lines)


def rag_pipeline(
    collection_name: str,
    embed_provider: str,
    llm_provider: str,
    llm_model_name: str,
    query: str,
    top_k: int = 3,
    score_threshold: float = 0.2,
    search_mode: str = "vector",
    candidate_k: Optional[int] = None,
    rerank_config: Optional[dict] = None,
    reference: Optional[str] = None,
    metadata_filter=None,
    auto_extract_filter: bool = True,
    run_eval=False,
    eval_model_name="gpt-4o-mini",
    eval_is_local=False,
    use_contextual: bool = False,
    use_multi_query: bool = False,
    multi_query_count: int = 5,
    multi_query_rrf_k: int = 60,
    conversation_history: list = None,
):
    if conversation_history is None:
        conversation_history = []

    langfuse = get_client()

    qdrant_filter = resolve_filter(
        query=query,
        explicit_filter=metadata_filter,
        auto_extract=auto_extract_filter,
    )

    with langfuse.start_as_current_observation(
        name="rag_pipeline",
        as_type="span",
        input=query,
        metadata={
            "collection_name": collection_name,
            "embed_provider": embed_provider,
            "llm_provider": llm_provider,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "score_threshold": score_threshold,
            "search_mode": search_mode,
            "rerank_config": rerank_config,
            "filter_applied": qdrant_filter is not None,
            "history_turns": len(conversation_history) // 2,
            "use_contextual": use_contextual,
            "use_multi_query": use_multi_query,
            "multi_query_count": multi_query_count,
            "multi_query_rrf_k": multi_query_rrf_k,
        },
    ) as pipeline_span:

        with langfuse.start_as_current_observation(
            name="retrieval",
            as_type="span",
            input={
                "query": query,
                "collection_name": collection_name,
                "embed_provider": embed_provider,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "score_threshold": score_threshold,
                "search_mode": search_mode,
                "rerank_config": rerank_config,
                "filter_applied": qdrant_filter is not None,
                "use_contextual": use_contextual,
                "use_multi_query": use_multi_query,
                "multi_query_count": multi_query_count,
                "multi_query_rrf_k": multi_query_rrf_k,
            },
        ) as retrieval_span:

            docs = retrieve(
                collection_name=collection_name,
                embed_provider=embed_provider,
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                search_mode=search_mode,
                query_filter=qdrant_filter,
                use_contextual=use_contextual,
                use_multi_query=use_multi_query,
                multi_query_count=multi_query_count,
                multi_query_rrf_k=multi_query_rrf_k,
                candidate_k=candidate_k,
                rerank_config=rerank_config,
            )

            retrieval_span.update(output=docs)

        with langfuse.start_as_current_observation(
            name="answer_generation",
            as_type="generation",
            model=llm_model_name,
            input={
                "query": query,
                "retrieved_docs": docs,
                "history_turns": len(conversation_history) // 2,
            }
        ) as generation:

            answer, usage = generate_answer(
                query,
                docs,
                provider=llm_provider,
                llm_model_name=llm_model_name,
                conversation_history=conversation_history,
            )

            source_verification = _verify_answer_sources(answer, docs)
            source_warning = _build_source_warning(source_verification)
            if source_warning:
                answer = f"{source_warning}\n\n{answer}"

            generation.update(
                output=answer,
                usage_details=usage,
            )

        updated_history = conversation_history + [
            {"role": "user",      "content": query},
            {"role": "assistant", "content": answer},
        ]

        result = {
            "user_input": query,
            "response": answer,
            "retrieved_context": [d.get("content", "") for d in docs],
            "reference": reference if reference is not None else find_reference_for_query(query),
            "updated_history": updated_history,
        }

        print("\n===== 답변 =====")
        print(answer)
        print("===============\n")

        if run_eval:

            print("--- [4] 평가 시작 ---")

            with langfuse.start_as_current_observation(
                name="ragas_evaluation",
                as_type="generation",
                model=eval_model_name,
                input=result,
            ) as generation:

                evaluate(
                    evaluation_data=[result],
                    model_name=eval_model_name,
                    is_local=eval_is_local,
                    langfuse=langfuse,
                    generation=generation,
                )

        pipeline_span.update(output=result)

    langfuse.flush()

    return result