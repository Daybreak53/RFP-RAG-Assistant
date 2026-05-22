from __future__ import annotations

import logging
from typing import Any, Optional

from omegaconf import DictConfig

from src.retrieval.filter_extractor import MetadataFilter
from src.generation.pipeline import rag_pipeline, find_reference_for_query

# 로거 설정
logger = logging.getLogger(__name__)

_HELP_TEXT = """
[특수 명령어 안내]
    - history | 히스토리 : 현재 기억 중인 대화 내역 출력
    - clear   | 초기화   : 대화 히스토리 초기화
    - exit    | quit     : 대화 종료
"""

_CMD_HISTORY = {"history", "히스토리"}
_CMD_CLEAR = {"clear", "초기화"}
_CMD_EXIT = {"exit", "quit"}


def _build_explicit_filter(cfg: DictConfig) -> MetadataFilter:
    """
    설정 객체에서 필터 조건을 추출하여 MetadataFilter 객체 생성
    """
    f = cfg.filter
    return MetadataFilter(
        organization=f.org,
        budget_min=f.budget_min,
        budget_max=f.budget_max,
        announcement_after=f.announce_after,
        announcement_before=f.announce_before,
        bid_start_after=f.bid_start_after,
        bid_deadline_before=f.bid_deadline_before,
        title_keyword=f.title,
        doc_id=f.doc_id,
    )


def _trim_history(history: list[dict[str, str]], max_turns: int) -> list[dict[str, str]]:
    """
    최대 기억 턴 수(max_turns)에 맞춰 대화 히스토리를 자르기
    (1턴 = 사용자 질문 + AI 답변 = 2개의 메시지)
    """
    max_messages = max_turns * 2
    return history[-max_messages:] if len(history) > max_messages else history


def _print_history(history: list[dict[str, str]]) -> None:
    """
    현재까지의 대화 히스토리를 콘솔에 출력
    """
    if not history:
        print("[알림] 현재 저장된 대화 히스토리가 없습니다.\n")
        return

    print("\n===== 📝 대화 히스토리 =====")
    for i in range(0, len(history), 2):
        turn = (i // 2) + 1
        user_msg = history[i].get("content", "")[:120].replace("\n", " ")
        
        assist_msg = ""
        if i + 1 < len(history):
            assist_msg = history[i + 1].get("content", "")[:120].replace("\n", " ")
            
        print(f"[{turn}턴]")
        print(f"  👤 사용자: {user_msg}...")
        print(f"  🤖 AI    : {assist_msg}...")
    print("============================\n")


def run_single_query(
    cfg: DictConfig,
    query_text: str,
    conversation_history: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    """
    단일 질문에 대해 RAG 파이프라인 실행
    """
    embed_provider = cfg.providers.embedding
    llm_provider = cfg.providers.llm
    collection_name = cfg.vector_db.collection_names[embed_provider]
    run_eval = cfg.pipeline.run_eval

    explicit_filter = _build_explicit_filter(cfg)
    auto_extract = not cfg.filter.no_auto

    # 평가 모드 활성화 시 reference(정답) 추출 및 검증
    reference = None
    if run_eval:
        reference = find_reference_for_query(query_text)
        if not reference:
            logger.error("평가 모드(--eval)는 평가 데이터셋에 있는 질의에만 사용할 수 있습니다.")
            raise SystemExit(
                "[평가 오류] data/eval_dataset.json의 user_input과 일치하는 질의가 아닙니다.\n"
                f"현재 질의: {query_text}"
            )

    # 파이프라인 실행 및 결과 반환
    return rag_pipeline(
        collection_name=collection_name,
        embed_provider=embed_provider,
        llm_provider=llm_provider,
        llm_model_name=cfg.providers.models.llm[llm_provider],
        query=query_text,
        top_k=cfg.retrieval.top_k,
        score_threshold=cfg.retrieval.score_threshold,
        search_mode=cfg.retrieval.search_mode,
        reference=reference,
        metadata_filter=explicit_filter,
        auto_extract_filter=auto_extract,
        run_eval=run_eval,
        eval_model_name=cfg.evaluation.model_name,
        eval_is_local=cfg.evaluation.is_local,
        use_contextual=cfg.parsing.use_contextual,
        conversation_history=conversation_history,
    )


def run_chat_mode(cfg: DictConfig) -> None:
    """
    CLI 기반의 대화형 모드 실행
    """
    max_turns = cfg.chat.max_history_turns

    print("\n" + "=" * 60)
    print(f"최대 기억 턴 수: {max_turns}")
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

        # 특수 명령어 처리
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

        # 질의 수행
        trimmed_history = _trim_history(conversation_history, max_turns)

        try:
            result = run_single_query(
                cfg=cfg,
                query_text=user_input,
                conversation_history=trimmed_history,
            )
            
            # 파이프라인으로부터 업데이트된 히스토리 반영
            conversation_history = result.get("updated_history", conversation_history)
            
            # 턴 수 계산 및 출력
            current_turns = len(conversation_history) // 2
            print(f"\n[💡 현재 대화 히스토리: {current_turns}턴 기억 중]\n")

        except Exception as e:
            logger.error("질의 처리 중 예기치 않은 오류가 발생했습니다.", exc_info=True)
            print(f"[오류] 질의 처리 중 문제가 발생했습니다: {e}\n")