from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from langfuse import get_client

from src.retrieval.retriever import retrieve
from src.retrieval.filter_extractor import resolve_filter, MetadataFilter
from src.generation.gen import generate_answer
from src.evaluation.evaluate import evaluate

# 로거 설정
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DATASET_PATHS = (PROJECT_ROOT / "data" / "eval_dataset.json",)


def _normalize_query(text: Optional[str]) -> str:
    """문자열 내의 불필요한 공백과 줄바꿈을 단일 공백으로 정규화"""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _load_reference_map() -> dict[str, str]:
    """
    평가용 데이터셋(JSON)에서 질의(user_input)와 정답(reference) 쌍을 로드하여 반환
    """
    reference_map: dict[str, str] = {}

    for dataset_path in EVAL_DATASET_PATHS:
        if not dataset_path.exists():
            logger.debug(f"평가 데이터셋을 찾을 수 없습니다: {dataset_path}")
            continue

        try:
            with dataset_path.open("r", encoding="utf-8") as f:
                records = json.load(f)

            for record in records:
                key = _normalize_query(record.get("user_input"))
                reference = record.get("reference")

                if key and reference and key not in reference_map:
                    reference_map[key] = reference
                    
        except json.JSONDecodeError as e:
            logger.error(f"평가 데이터셋 JSON 파싱 오류 ({dataset_path}): {e}")
        except Exception as e:
            logger.error(f"평가 데이터셋 로드 중 알 수 없는 오류 발생: {e}")

    return reference_map


def find_reference_for_query(query: str) -> str:
    """
    사용자 질의에 대응하는 정답(reference)을 평가 데이터셋에서 탐색
    """
    reference_map = _load_reference_map()
    normalized_query = _normalize_query(query)

    # 1차 시도: 정규화된 텍스트 완전 일치
    if normalized_query in reference_map:
        return reference_map[normalized_query]

    # 2차 시도: 모든 공백을 제거한 상태로 일치 검사
    compact_query = re.sub(r"\s+", "", normalized_query)
    for eval_query, reference in reference_map.items():
        if re.sub(r"\s+", "", eval_query) == compact_query:
            return reference

    return ""


def rag_pipeline(
    collection_name: str,
    embed_provider: str,
    llm_provider: str,
    llm_model_name: str,
    query: str,
    top_k: int = 3,
    score_threshold: float = 0.2,
    search_mode: str = "vector",
    reference: Optional[str] = None,
    metadata_filter: Optional[MetadataFilter] = None,
    auto_extract_filter: bool = True,
    run_eval: bool = False,
    eval_model_name: str = "gpt-5-nano",
    eval_is_local: bool = False,
    use_contextual: bool = False,
    use_multi_query: bool = False,
    multi_query_count: int = 5,
    conversation_history: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    """
    주어진 설정과 질의를 바탕으로 검색, 답변 생성, (선택적) 평가를 수행하는 메인 RAG 파이프라인
    """
    history = conversation_history or []
    langfuse = get_client()

    # 1. 자연어 질의 및 명시적 설정 기반의 메타데이터 필터 추출
    qdrant_filter = resolve_filter(
        query=query,
        explicit_filter=metadata_filter,
        auto_extract=auto_extract_filter,
    )

    # 2. Langfuse 파이프라인 관측 시작
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
            "search_mode": search_mode,
            "filter_applied": qdrant_filter is not None,
            "history_turns": len(history) // 2,
            "use_contextual": use_contextual,
            "use_multi_query": use_multi_query,
            "multi_query_count": multi_query_count,
        },
    ) as pipeline_span:

        # -------------------------------------------------------------
        # 2-1. 문서 검색 (Retrieval)
        # -------------------------------------------------------------
        with langfuse.start_as_current_observation(
            name="retrieval",
            as_type="span",
            input={
                "query": query,
                "collection_name": collection_name,
                "embed_provider": embed_provider,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "search_mode": search_mode,
                "filter_applied": qdrant_filter is not None,
                "use_contextual": use_contextual,
                "use_multi_query": use_multi_query,
                "multi_query_count": multi_query_count,
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
            )
            retrieval_span.update(output=docs)

        # -------------------------------------------------------------
        # 2-2. 답변 생성 (Generation)
        # -------------------------------------------------------------
        with langfuse.start_as_current_observation(
            name="answer_generation",
            as_type="generation",
            model=llm_model_name,
            input={
                "query": query,
                "retrieved_docs": docs,
                "history_turns": len(history) // 2,
            }
        ) as generation_span:
            
            answer, usage = generate_answer(
                query=query,
                docs=docs,
                provider=llm_provider,
                llm_model_name=llm_model_name,
                conversation_history=history,
            )
            
            generation_span.update(
                output=answer,
                usage_details=usage,
            )

        # 히스토리 업데이트
        updated_history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]

        # 파이프라인 반환 결과 구성
        result = {
            "user_input": query,
            "response": answer,
            "retrieved_context": [d.get("content", "") for d in docs],
            "reference": reference if reference is not None else find_reference_for_query(query),
            "updated_history": updated_history,
        }

        # 사용자 콘솔 출력용 UI
        print("\n===== 🤖 답변 =====")
        print(answer)
        print("===================\n")

        # -------------------------------------------------------------
        # 2-3. 파이프라인 자동 평가 (Evaluation)
        # -------------------------------------------------------------
        if run_eval:
            logger.info("--- [4] 파이프라인 평가(RAGAS) 시작 ---")
            
            with langfuse.start_as_current_observation(
                name="ragas_evaluation",
                as_type="generation",
                model=eval_model_name,
                input=result,
            ) as eval_span:
                
                evaluate(
                    evaluation_data=[result],
                    model_name=eval_model_name,
                    is_local=eval_is_local,
                    langfuse=langfuse,
                    generation=eval_span,
                )

        pipeline_span.update(output=result)

    # Langfuse 데이터 전송
    langfuse.flush()

    return result