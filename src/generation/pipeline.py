from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from langfuse import get_client

from src.retrieval.retriever import retrieve
from src.retrieval.filter_extractor import resolve_filter, MetadataFilter
from src.retrieval.query_router import route, RouteConfig
from src.generation.gen import generate_answer, generate_pure_text
from src.evaluation.evaluate import evaluate

# 로거 설정
logger = logging.getLogger(__name__)

PROJECT_ROOT       = Path(__file__).resolve().parents[2]
EVAL_DATASET_PATHS = (PROJECT_ROOT / "data" / "eval_dataset.json",)


def _normalize_query(text: Optional[str]) -> str:
    """
    문자열 내의 불필요한 공백과 줄바꿈을 단일 공백으로 정규화
    """
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _load_reference_map() -> dict[str, str]:
    """
    평가용 데이터셋(JSON)에서 질의(user_input)와 정답(reference) 쌍을 로드하여 반환
    """
    reference_map: dict[str, str] = {}
    for dataset_path in EVAL_DATASET_PATHS:
        if not dataset_path.exists():
            continue
        try:
            with dataset_path.open("r", encoding="utf-8") as f:
                records = json.load(f)
            for record in records:
                key = _normalize_query(record.get("user_input"))
                ref = record.get("reference")
                if key and ref and key not in reference_map:
                    reference_map[key] = ref
        except json.JSONDecodeError as e:
            logger.error(f"평가 데이터셋 파싱 오류: {e}")
        except Exception as e:
            logger.error(f"평가 데이터셋 로드 오류: {e}")
    return reference_map


def _normalize_spaces(text: str) -> str:
    """
    비교용 공백 제거 정규화
    """
    return re.sub(r"\s+", "", text)


ambiguous_markers = [
        "첫 번째", "두 번째", "세 번째", "네 번째", "다섯 번째",
        "첫째", "둘째", "셋째",
        "1번째", "2번째", "3번째",
        "앞서", "위의", "위에서", "아래의", "이전",
        "방금", "아까", "직전", "바로 전",

        "다시", "재설명", "한 번 더", "좀 더", "더 자세히",

        "그 사업", "이 사업", "해당 사업", "본 사업", "동 사업",
        "그 용역", "이 용역", "해당 용역",
        "그 RFP", "이 RFP", "해당 RFP",
        "그 공고", "이 공고", "해당 공고",
        "그 문서", "이 문서", "해당 문서",
        "그 제안요청서", "해당 제안요청서",

        "그 기관", "이 기관", "해당 기관",
        "그 발주처", "해당 발주처",
        "거기서", "거기에서",

        "그 내용", "이 내용", "해당 내용",
        "그 조건", "이 조건", "해당 조건",
        "그 금액", "이 금액", "해당 금액",
        "그 예산", "이 예산", "해당 예산",
        "그 일정", "이 일정", "해당 일정",
        "그 결과", "이 결과", "해당 결과",
        "그 항목", "이 항목", "해당 항목",
        "그 기준", "이 기준", "해당 기준",

        "그럼", "그렇다면", "그렇면",
        "그래서", "따라서", "그 경우",
        "다른 건", "다른 것은"
    ]

_AMBIGUOUS_MARKERS_NORMALIZED = [_normalize_spaces(m) for m in ambiguous_markers]


def _is_ambiguous(query: str, normalized_markers: list[str]) -> bool:
    query_no_space = _normalize_spaces(query)
    for marker_no_space in normalized_markers:
        if marker_no_space in query_no_space:
            return True
    return False


def _rewrite_query_with_history(
    query: str,
    history: list[dict[str, str]],
    provider: str,
    model: str,
) -> str:
    """
    대화 히스토리를 참고하여 모호한 쿼리를 독립적인 검색 쿼리로 재작성
    """
    if not history:
        return query

    if not _is_ambiguous(query, _AMBIGUOUS_MARKERS_NORMALIZED):
        return query

    history_text = "\n".join(
        f"{'사용자' if m['role'] == 'user' else 'AI'}: {m['content'][:200]}"
        for m in history[-4:]  # 최근 2턴만 참고
    )
    prompt = f"""아래 대화 히스토리를 참고하여, 마지막 질문을 재작성하세요.

[재작성 규칙]
1. "그", "해당", "이", "그것", "거기", "첫 번째", "두 번째" 등 모든 지시어와 대명사를 히스토리에서 찾은 구체적인 고유명사(사업명, 기관명 등)로 반드시 교체하세요.
2. 재작성된 문장에 지시어나 대명사가 하나도 남아있으면 안 됩니다.
3. 재작성된 질문만 출력하고 다른 설명은 하지 마세요.

[올바른 예시]
- 원본: "그 사업의 예산은?" → 재작성: "서울시 지도정보 플랫폼 고도화 용역 사업의 예산은?"
- 원본: "두 번째 답을 다시 설명해줘" → 재작성: "서울시 지도정보 플랫폼 사업의 예산을 다시 설명해줘"

[대화 히스토리]
{history_text}

[재작성할 질문]
{query}

[재작성 결과]"""

    try:
        rewritten = generate_pure_text(prompt, provider=provider, llm_model_name=model)
        rewritten = rewritten.strip()
        logger.info(f"[QueryRewrite] '{query}' → '{rewritten}'")
        return rewritten if rewritten else query
    except Exception as e:
        logger.warning(f"쿼리 재작성 실패, 원본 사용: {e}")
        return query


def find_reference_for_query(query: str) -> str:
    """
    사용자 질의에 대응하는 정답(reference)을 평가 데이터셋에서 탐색
    """
    reference_map   = _load_reference_map()
    normalized      = _normalize_query(query)
    if normalized in reference_map:
        return reference_map[normalized]
    compact = re.sub(r"\s+", "", normalized)
    for k, v in reference_map.items():
        if re.sub(r"\s+", "", k) == compact:
            return v
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
    collection_name:   str,
    embed_provider:    str,
    llm_provider:      str,
    llm_model_name:    str,
    query:             str,
    top_k:             int   = 3,
    score_threshold:   float = 0.2,
    search_mode:       str   = "vector",
    candidate_k:       Optional[int]  = None,
    rerank_config:     Optional[dict] = None,
    reference:         Optional[str] = None,
    metadata_filter:   Optional[MetadataFilter] = None,
    auto_extract_filter: bool = True,
    run_eval:          bool  = False,
    eval_model_name:   str   = "gpt-5-nano",
    eval_is_local:     bool  = False,
    use_contextual:    bool  = False,
    use_multi_query:   bool  = False,
    multi_query_count: int   = 5,
    multi_query_rrf_k: int   = 60,
    conversation_history: Optional[list[dict[str, str]]] = None,
    use_query_router:    bool          = True,
    use_llm_classifier:  bool          = False,
    router_cfg:          Optional[dict] = None,
    force_query_type:    Optional[str]  = None,
    use_query_rewrite: bool = False,
) -> dict[str, Any]:
    """
    주어진 설정과 질의를 바탕으로 검색, 답변 생성, (선택적) 평가를 수행하는 메인 RAG 파이프라인
    """
    history = conversation_history or []
    langfuse = get_client()

    # ──────────────────────────────────────────
    # Step 0. QueryRouter
    # ──────────────────────────────────────────
    route_cfg: Optional[RouteConfig] = None

    if use_query_router:
        with langfuse.start_as_current_observation(
            name="query_routing",
            as_type="span",
            input={"query": query, "use_llm_classifier": use_llm_classifier},
        ) as router_span:

            route_cfg = route(
                query               = query,
                use_llm_classifier  = use_llm_classifier,
                llm_provider        = llm_provider,
                llm_model           = llm_model_name,
                router_cfg          = router_cfg,
                force_query_type    = force_query_type,
            )

            router_span.update(output={
                "query_type":      route_cfg.query_type.value,
                "search_mode":     route_cfg.search_mode,
                "top_k":           route_cfg.top_k,
                "score_threshold": route_cfg.score_threshold,
                "prompt_mode":     route_cfg.prompt_mode.value,
                "use_multi_query": route_cfg.use_multi_query,
                "reason":          route_cfg.reason,
            })

        # RouteConfig로 검색 파라미터 override
        effective_search_mode   = route_cfg.search_mode
        effective_top_k         = route_cfg.top_k
        effective_threshold     = route_cfg.score_threshold
        effective_prompt_mode   = route_cfg.prompt_mode.value
        effective_query_type    = route_cfg.query_type.value
        effective_multi_query   = route_cfg.use_multi_query
        effective_llm_provider  = route_cfg.llm_provider  or llm_provider
        effective_llm_model     = route_cfg.llm_model     or llm_model_name

        logger.info(
            f"[Router] type={effective_query_type} | "
            f"search={effective_search_mode} | top_k={effective_top_k} | "
            f"threshold={effective_threshold} | prompt={effective_prompt_mode} | "
            f"multi_query={effective_multi_query} | reason={route_cfg.reason}"
        )

    else:
        effective_search_mode  = search_mode
        effective_top_k        = top_k
        effective_threshold    = score_threshold
        effective_prompt_mode  = "basic"
        effective_query_type   = "unknown"
        effective_multi_query  = use_multi_query
        effective_llm_provider = llm_provider
        effective_llm_model    = llm_model_name
        logger.info("[Router] 비활성화 - config 기본값 사용")
        logger.info(
            f"[Retriever] search_mode={effective_search_mode} | "
            f"top_k={effective_top_k} | "
            f"threshold={effective_threshold}"
        )

    # ──────────────────────────────────────────
    # Step 1. 메타데이터 필터 추출
    # ──────────────────────────────────────────
    qdrant_filter = resolve_filter(
        query            = query,
        explicit_filter  = metadata_filter,
        auto_extract     = auto_extract_filter,
        query_type       = effective_query_type,
    )

    # ──────────────────────────────────────────
    # Langfuse 파이프라인 span 시작
    # ──────────────────────────────────────────
    with langfuse.start_as_current_observation(
        name     = "rag_pipeline",
        as_type  = "span",
        input    = query,
        metadata = {
            "collection_name":   collection_name,
            "embed_provider":    embed_provider,
            "llm_provider":      effective_llm_provider,
            "top_k":             effective_top_k,
            "candidate_k":       candidate_k,
            "score_threshold":   effective_threshold,
            "search_mode":       effective_search_mode,
            "rerank_config":     rerank_config,
            "prompt_mode":       effective_prompt_mode,
            "query_type":        effective_query_type,
            "filter_applied":    qdrant_filter is not None,
            "history_turns":     len(history) // 2,
            "use_contextual":    use_contextual,
            "use_multi_query":   effective_multi_query,
            "multi_query_count": multi_query_count,
            "multi_query_rrf_k": multi_query_rrf_k,
            "router_active":     use_query_router,
        },
    ) as pipeline_span:

        # ──────────────────────────────────────
        # Step 2. 문서 검색
        # ──────────────────────────────────────
        effective_query = query
        if use_query_rewrite and history:
            effective_query = _rewrite_query_with_history(
                query, history, effective_llm_provider, effective_llm_model
            )

        with langfuse.start_as_current_observation(
            name    = "retrieval",
            as_type = "span",
            input   = {
                "query":             query,
                "search_mode":       effective_search_mode,
                "top_k":             effective_top_k,
                "candidate_k":       candidate_k,
                "score_threshold":   effective_threshold,
                "use_multi_query":   effective_multi_query,
                "multi_query_count": multi_query_count,
                "multi_query_rrf_k": multi_query_rrf_k,
                "rerank_config":     rerank_config,
                "filter_applied":    qdrant_filter is not None,
            },
        ) as retrieval_span:

            docs = retrieve(
                collection_name   = collection_name,
                embed_provider    = embed_provider,
                query             = effective_query,
                top_k             = effective_top_k,
                score_threshold   = effective_threshold,
                search_mode       = effective_search_mode,
                query_filter      = qdrant_filter,
                use_contextual    = use_contextual,
                use_multi_query   = effective_multi_query,
                multi_query_count = multi_query_count,
                multi_query_rrf_k = multi_query_rrf_k,
                candidate_k       = candidate_k,
                rerank_config     = rerank_config,
            )
            retrieval_span.update(output=docs)

        # ──────────────────────────────────────
        # Step 3. 답변 생성
        # ──────────────────────────────────────
        with langfuse.start_as_current_observation(
            name    = "answer_generation",
            as_type = "generation",
            model   = effective_llm_model,
            input   = {
                "query":         query,
                "retrieved_docs": docs,
                "history_turns": len(history) // 2,
                "prompt_mode":   effective_prompt_mode,
                "query_type":    effective_query_type,
            },
        ) as generation_span:

            answer, usage = generate_answer(
                query                = query,
                docs                 = docs,
                provider             = effective_llm_provider,
                llm_model_name       = effective_llm_model,
                conversation_history = history,
                prompt_mode          = effective_prompt_mode,   # ← 신규
                query_type           = effective_query_type,    # ← 신규
            )

            source_verification = _verify_answer_sources(answer, docs)
            source_warning      = _build_source_warning(source_verification)
            if source_warning:
                answer = f"{source_warning}\n\n{answer}"

            generation_span.update(
                output        = answer,
                usage_details = usage,
            )

        # 히스토리 업데이트
        updated_history = history + [
            {"role": "user",      "content": query},
            {"role": "assistant", "content": answer},
        ]

        result = {
            "user_input":         query,
            "response":           answer,
            "retrieved_context":  [d.get("content", "") for d in docs],
            "reference":          reference if reference is not None else find_reference_for_query(query),
            "updated_history":    updated_history,
            "routing": {
                "query_type":    effective_query_type,
                "search_mode":   effective_search_mode,
                "prompt_mode":   effective_prompt_mode,
                "top_k":         effective_top_k,
                "use_multi_query": effective_multi_query,
            },
        }

        # 콘솔 출력
        print("\n===== 🤖 답변 =====")
        print(answer)
        print("===================\n")

        # ──────────────────────────────────────
        # Step 4. RAGAS 평가 (선택)
        # ──────────────────────────────────────
        if run_eval:
            logger.info("--- [4] 파이프라인 평가(RAGAS) 시작 ---")
            with langfuse.start_as_current_observation(
                name    = "ragas_evaluation",
                as_type = "generation",
                model   = eval_model_name,
                input   = result,
            ) as eval_span:
                evaluate(
                    evaluation_data = [result],
                    model_name      = eval_model_name,
                    is_local        = eval_is_local,
                    langfuse        = langfuse,
                    generation      = eval_span,
                )

        pipeline_span.update(output=result)

    langfuse.flush()
    return result