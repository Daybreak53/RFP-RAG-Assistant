from __future__ import annotations

import logging
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from src.retrieval.filter_extractor import MetadataFilter
from src.generation.pipeline import rag_pipeline, find_reference_for_query

logger = logging.getLogger(__name__)

_HELP_TEXT = """
[특수 명령어 안내]
    - history | 히스토리 : 현재 기억 중인 대화 내역 출력
    - clear   | 초기화   : 대화 히스토리 초기화
    - exit    | quit     : 대화 종료
"""

_CMD_HISTORY = {"history", "히스토리"}
_CMD_CLEAR   = {"clear",   "초기화"}
_CMD_EXIT    = {"exit",    "quit"}


def _build_explicit_filter(cfg: DictConfig) -> MetadataFilter:
    """
    설정 객체에서 필터 조건을 추출하여 MetadataFilter 객체 생성
    """
    f = cfg.filter
    return MetadataFilter(
        organization       = f.org,
        budget_min         = f.budget_min,
        budget_max         = f.budget_max,
        announcement_after = f.announce_after,
        announcement_before= f.announce_before,
        bid_start_after    = f.bid_start_after,
        bid_deadline_before= f.bid_deadline_before,
        title_keyword      = f.title,
        doc_id             = f.doc_id,
    )


def _build_router_cfg(cfg: DictConfig) -> Optional[dict]:
    """
    config.yaml의 router 섹션을 순수 dict로 변환
    """
    if not hasattr(cfg, "router"):
        return None
    return OmegaConf.to_container(cfg.router, resolve=True)


def _trim_history(
    history: list[dict[str, str]], 
    max_turns: int,
) -> list[dict[str, str]]:
    max_messages = max_turns * 2
    return history[-max_messages:] if len(history) > max_messages else history


def _print_history(history: list[dict[str, str]]) -> None:
    if not history:
        print("[알림] 현재 저장된 대화 히스토리가 없습니다.\n")
        return
    print("\n===== 📝 대화 히스토리 =====")
    for i in range(0, len(history), 2):
        turn = (i // 2) + 1
        user_msg   = history[i].get("content", "")[:120].replace("\n", " ")
        assist_msg = ""
        if i + 1 < len(history):
            assist_msg = history[i + 1].get("content", "")[:120].replace("\n", " ")
        print(f"[{turn}턴]\n  👤 사용자: {user_msg}...\n  🤖 AI    : {assist_msg}...")
    print("============================\n")


def run_single_query(
    cfg: DictConfig,
    query_text: str,
    conversation_history: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    """
    단일 질문에 대해 RAG 파이프라인 실행
    """

    embed_provider  = cfg.providers.embedding
    llm_provider    = cfg.providers.llm
    collection_name = cfg.vector_db.collection_names[embed_provider]
    run_eval        = cfg.pipeline.run_eval

    explicit_filter = _build_explicit_filter(cfg)
    auto_extract    = not cfg.filter.no_auto

    # Router 설정 추출
    router_section   = _build_router_cfg(cfg)
    use_router       = router_section.get("enabled", True)       if router_section else True
    use_llm_cls      = router_section.get("use_llm_classifier", False) if router_section else False
    force_type       = router_section.get("force_query_type")    if router_section else None

    # 평가 모드: reference(정답) 확인
    reference = None
    if run_eval:
        reference = find_reference_for_query(query_text)
        if not reference:
            logger.error("평가 모드는 eval_dataset.json에 등록된 질의에만 사용 가능합니다.")
            raise SystemExit(
                f"[평가 오류] 일치하는 질의가 없습니다.\n현재 질의: {query_text}"
            )

    return rag_pipeline(
        collection_name       = collection_name,
        embed_provider        = embed_provider,
        llm_provider          = llm_provider,
        llm_model_name        = cfg.providers.models.llm[llm_provider],
        query                 = query_text,
        top_k                 = cfg.retrieval.top_k,
        score_threshold       = cfg.retrieval.score_threshold,
        search_mode           = cfg.retrieval.search_mode,
        reference             = reference,
        metadata_filter       = explicit_filter,
        auto_extract_filter   = auto_extract,
        run_eval              = run_eval,
        eval_model_name       = cfg.evaluation.model_name,
        eval_is_local         = cfg.evaluation.is_local,
        use_contextual        = cfg.parsing.use_contextual,
        conversation_history  = conversation_history,
        use_query_router      = use_router,
        use_llm_classifier    = use_llm_cls,
        router_cfg            = router_section,
        force_query_type      = force_type,
    )


def run_chat_mode(cfg: DictConfig) -> None:
    """
    CLI 기반 대화형 모드 실행
    """
    max_turns = cfg.chat.max_history_turns

    # 라우터 활성화 여부 안내
    router_section = _build_router_cfg(cfg)
    router_enabled = router_section.get("enabled", True) if router_section else True
    router_mode    = (
        "LLM 분류기" if (router_section or {}).get("use_llm_classifier") else "규칙 기반 분류기"
    ) if router_enabled else "비활성화"

    print("\n" + "=" * 60)
    print(f"최대 기억 턴 수: {max_turns}")
    print(f"QueryRouter: {router_mode}")
    print(_HELP_TEXT)
    print("=" * 60 + "\n")

    conversation_history: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("👤 질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[알림] 대화를 강제 종료합니다.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in _CMD_EXIT:
            print("[알림] 대화를 종료합니다. 이용해 주셔서 감사합니다.")
            break

        if cmd in _CMD_HISTORY:
            _print_history(conversation_history)
            continue

        if cmd in _CMD_CLEAR:
            conversation_history.clear()
            print("[알림] 대화 히스토리가 초기화되었습니다.\n")
            continue

        trimmed_history = _trim_history(conversation_history, max_turns)

        try:
            result = run_single_query(
                cfg                  = cfg,
                query_text           = user_input,
                conversation_history = trimmed_history,
            )
            conversation_history = result.get("updated_history", conversation_history)

            # 라우팅 결과 표시
            routing = result.get("routing", {})
            current_turns = len(conversation_history) // 2
            print(
                f"[💡 히스토리: {current_turns}턴 | "
                f"유형: {routing.get('query_type','?')} | "
                f"검색: {routing.get('search_mode','?')} | "
                f"프롬프트: {routing.get('prompt_mode','?')}]\n"
            )

        except Exception as e:
            logger.error("질의 처리 중 예기치 않은 오류", exc_info=True)
            print(f"[오류] 질의 처리 중 문제가 발생했습니다: {e}\n")