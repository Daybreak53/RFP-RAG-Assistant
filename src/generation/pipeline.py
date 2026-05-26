from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from langfuse import get_client

from src.retrieval.retriever import retrieve
from src.retrieval.filter_extractor import resolve_filter, MetadataFilter
from src.retrieval.query_router import route, RouteConfig, QueryType, PromptMode
from src.generation.gen import generate_answer
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


def rag_pipeline(
    collection_name:   str,
    embed_provider:    str,
    llm_provider:      str,
    llm_model_name:    str,
    query:             str,
    top_k:             int   = 3,
    score_threshold:   float = 0.2,
    search_mode:       str   = "vector",
    reference:         Optional[str] = None,
    metadata_filter:   Optional[MetadataFilter] = None,
    auto_extract_filter: bool = True,
    run_eval:          bool  = False,
    eval_model_name:   str   = "gpt-5-nano",
    eval_is_local:     bool  = False,
    use_contextual:    bool  = False,
    use_multi_query:   bool  = False,
    multi_query_count: int   = 5,
    conversation_history: Optional[list[dict[str, str]]] = None,
    use_query_router:    bool          = True,
    use_llm_classifier:  bool          = False,
    router_cfg:          Optional[dict] = None,
    force_query_type:    Optional[str]  = None,
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
        logger.info("[Router] 비활성화 — config 기본값 사용")

    # ──────────────────────────────────────────
    # Step 1. 메타데이터 필터 추출
    # ──────────────────────────────────────────
    qdrant_filter = resolve_filter(
        query            = query,
        explicit_filter  = metadata_filter,
        auto_extract     = auto_extract_filter,
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
            "score_threshold":   effective_threshold,
            "search_mode":       effective_search_mode,
            "prompt_mode":       effective_prompt_mode,
            "query_type":        effective_query_type,
            "filter_applied":    qdrant_filter is not None,
            "history_turns":     len(history) // 2,
            "use_contextual":    use_contextual,
            "use_multi_query":   effective_multi_query,
            "multi_query_count": multi_query_count,
            "router_active":     use_query_router,
        },
    ) as pipeline_span:

        # ──────────────────────────────────────
        # Step 2. 문서 검색
        # ──────────────────────────────────────
        with langfuse.start_as_current_observation(
            name    = "retrieval",
            as_type = "span",
            input   = {
                "query":           query,
                "search_mode":     effective_search_mode,
                "top_k":           effective_top_k,
                "score_threshold": effective_threshold,
                "use_multi_query": effective_multi_query,
            },
        ) as retrieval_span:

            docs = retrieve(
                collection_name  = collection_name,
                embed_provider   = embed_provider,
                query            = query,
                top_k            = effective_top_k,
                score_threshold  = effective_threshold,
                search_mode      = effective_search_mode,
                query_filter     = qdrant_filter,
                use_contextual   = use_contextual,
                use_multi_query  = effective_multi_query,
                multi_query_count= multi_query_count,
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
        print(f"[질의 유형: {effective_query_type} | 검색: {effective_search_mode} | "
                f"프롬프트: {effective_prompt_mode}]")
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