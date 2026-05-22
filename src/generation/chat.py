from __future__ import annotations

from omegaconf import DictConfig
from src.retrieval.filter_extractor import MetadataFilter
from src.generation.pipeline import rag_pipeline, find_reference_for_query


def _build_explicit_filter(cfg: DictConfig) -> MetadataFilter:
    f = cfg.filter
    return MetadataFilter(
        organization        = f.org,
        budget_min          = f.budget_min,
        budget_max          = f.budget_max,
        announcement_after  = f.announce_after,
        announcement_before = f.announce_before,
        bid_start_after     = f.bid_start_after,
        bid_deadline_before = f.bid_deadline_before,
        title_keyword       = f.title,
        doc_id              = f.doc_id,
    )


def _trim_history(history: list[dict], max_turns: int) -> list[dict]:
    max_messages = max_turns * 2
    return history[-max_messages:] if len(history) > max_messages else history


def _print_history(history: list[dict]) -> None:
    if not history:
        print("[히스토리 없음]\n")
        return

    print("\n===== 대화 히스토리 =====")
    for i in range(0, len(history), 2):
        turn = i // 2 + 1
        user_msg    = history[i]["content"][:120].replace("\n", " ")
        assist_msg  = history[i + 1]["content"][:120].replace("\n", " ") if i + 1 < len(history) else ""
        print(f"[{turn}턴]")
        print(f"  👤 {user_msg}...")
        print(f"  🤖 {assist_msg}...")
    print("=========================\n")


# 단일 질의 실행
def run_single_query(
    cfg: DictConfig,
    query_text: str,
    conversation_history: list[dict] | None = None,
) -> dict:

    embed_provider  = cfg.providers.embedding
    llm_provider    = cfg.providers.llm
    collection_name = cfg.collection_name[embed_provider]
    run_eval        = cfg.run_eval

    explicit_filter = _build_explicit_filter(cfg)
    auto_extract    = not cfg.filter.no_auto

    reference = None
    if run_eval:
        reference = find_reference_for_query(query_text)
        if not reference:
            raise SystemExit(
                "[평가 오류] --eval은 data/eval_dataset.json의 user_input과 "
                "매칭되는 질의에서만 사용할 수 있습니다.\n"
                f"현재 질의: {query_text}"
            )

    return rag_pipeline(
        collection_name     = collection_name,
        embed_provider      = embed_provider,
        llm_provider        = llm_provider,
        llm_model_name      = cfg.llm_model_name[llm_provider],
        query               = query_text,
        top_k               = cfg.retrieval.top_k,
        score_threshold     = cfg.retrieval.score_threshold,
        search_mode         = cfg.retrieval.search_mode,
        reference           = reference,
        metadata_filter     = explicit_filter,
        auto_extract_filter = auto_extract,
        run_eval            = run_eval,
        eval_model_name     = cfg.evaluation.model_name,
        eval_is_local       = cfg.evaluation.is_local,
        use_contextual      = cfg.parsing.use_contextual,
        conversation_history= conversation_history,
    )


# 대화형 루프
_HELP_TEXT = """
특수 명령어:
  history | 히스토리  현재 기억 중인 대화 내역 출력
  clear   | 초기화    대화 히스토리 초기화
  exit    | quit      대화 종료
"""

_CMD_HISTORY = {"history", "히스토리"}
_CMD_CLEAR   = {"clear", "초기화"}
_CMD_EXIT    = {"exit", "quit"}


def run_chat_mode(cfg: DictConfig) -> None:
    max_turns = cfg.chat.max_history_turns

    print("\n" + "=" * 60)
    print(f"대화 기억 최대 턴 수: {max_turns}")
    print(_HELP_TEXT)
    print("=" * 60 + "\n")

    conversation_history: list[dict] = []

    while True:
        # 입력 수신
        try:
            user_input = input("질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[대화 종료]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # 특수 명령어 처리
        if cmd in _CMD_EXIT:
            print("[대화 종료]")
            break

        if cmd in _CMD_HISTORY:
            _print_history(conversation_history)
            continue

        if cmd in _CMD_CLEAR:
            conversation_history = []
            print("[대화 히스토리가 초기화되었습니다.]\n")
            continue

        # 질의 실행
        trimmed = _trim_history(conversation_history, max_turns)

        try:
            result = run_single_query(
                cfg=cfg,
                query_text=user_input,
                conversation_history=trimmed,
            )
        except Exception as e:
            print(f"[오류] 질의 처리 중 문제가 발생했습니다: {e}\n")
            continue

        conversation_history = result.get("updated_history", conversation_history)
        print(f"[현재 대화 히스토리: {len(conversation_history) // 2}턴 기억 중]\n")