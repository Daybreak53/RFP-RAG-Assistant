from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    FACTUAL       = "factual"        # 사실 조회형
    COMPARATIVE   = "comparative"    # 비교 분석형
    SUMMARIZATION = "summarization"  # 요약형
    PROCEDURAL    = "procedural"     # 절차·방법형
    ANALYTICAL    = "analytical"     # 분석·평가형
    FILTER_BASED  = "filter_based"   # 필터 조건형
    UNKNOWN       = "unknown"        # 분류 불가 → 기본값 적용


class PromptMode(str, Enum):
    BASIC    = "basic"     # 기본 4규칙 프롬프트
    COT_ZERO = "cot_zero"  # Zero-shot Chain-of-Thought
    COT_FEW  = "cot_few"   # Few-shot Chain-of-Thought


@dataclass
class RouteConfig:
    """
    라우터가 결정한 검색·생성 파라미터 집합
    """

    query_type:      QueryType
    search_mode:     str              # vector | keyword | hybrid | mmr | hyde
    top_k:           int
    score_threshold: float
    prompt_mode:     PromptMode

    # None이면 config에서 지정된 기본 모델·제공자 사용
    llm_provider:    Optional[str] = None
    llm_model:       Optional[str] = None

    use_multi_query: bool = False
    reason:          str  = ""       # 분류 근거

    def describe(self) -> str:
        return (
            f"[QueryRouter] type={self.query_type.value} | "
            f"search={self.search_mode} | top_k={self.top_k} | "
            f"threshold={self.score_threshold} | prompt={self.prompt_mode.value} | "
            f"multi_query={self.use_multi_query}"
        )


# 각 유형을 나타내는 한국어 키워드 정규식
_RULES: list[tuple[QueryType, re.Pattern]] = [
    # FILTER_BASED : 기관명·예산·날짜 조건이 포함된 질의
    (QueryType.FILTER_BASED, re.compile(
        r"발주\s*기관|발주처|수요\s*기관|기관명|"
        r"\d+\s*억|예산|사업비|금액|"
        r"\d{4}[-./년]\s*\d{1,2}[-./월]\s*\d{1,2}|"
        r"공고일|마감일|입찰\s*기간|일정"
    )),

    # FACTUAL : 수치·제한·조건 등 단일 사실 조회
    (QueryType.FACTUAL, re.compile(
        r"얼마|몇\s*(?:페이지|점|개|부|명|일|주|달|년|건|%)|"
        r"최대|최소|상한|하한|제한|기준|조건|몇\s*자|"
        r"언제|날짜|기간|몇\s*점|합격|배점|가점|가중치"
    )),

    # COMPARATIVE : 두 개 이상의 항목 비교
    (QueryType.COMPARATIVE, re.compile(
        r"비교|차이|대비|각각|어떻게\s*다|반면|"
        r"vs\.|versus|공통점|차이점|더\s*높|더\s*낮|"
        r"두\s*(?:사업|기관|문서|공고)|여러\s*(?:사업|기관)"
    )),

    # PROCEDURAL : 절차·방법·단계
    (QueryType.PROCEDURAL, re.compile(
        r"방법|절차|단계|어떻게|어떤\s*방식|"
        r"어디에|제출|접수|신청|참여|등록|"
        r"제안서\s*작성|준비|요구\s*사항|필수|"
        r"유의\s*사항|주의"
    )),

    # SUMMARIZATION : 요약·설명·개요
    (QueryType.SUMMARIZATION, re.compile(
        r"요약|정리|요점|개요|전반적|전체적|주요\s*내용|"
        r"핵심|알려줘|설명해|소개|어떤\s*사업|"
        r"무슨\s*사업|어떤\s*내용"
    )),

    # ANALYTICAL : 평가·분석·전략
    (QueryType.ANALYTICAL, re.compile(
        r"분석|평가|검토|적합|왜|이유|원인|"
        r"장점|단점|장단점|강점|약점|문제점|"
        r"전략|방향|고려|판단|타당|의미|영향"
    )),
]


def _classify_rule_based(query: str) -> tuple[QueryType, str]:
    """
    키워드 패턴 매칭으로 질의 유형 판별. 복수 매칭 시 첫 번째 우선
    """
    for qtype, pattern in _RULES:
        m = pattern.search(query)
        if m:
            return qtype, f"규칙 매칭: '{m.group()}'"
    return QueryType.UNKNOWN, "규칙 미매칭"


_LLM_CLASSIFY_PROMPT = """\
당신은 한국 공공입찰 RFP 문서 검색 시스템의 질의 분류 전문가입니다.
사용자의 질의를 분석하여 아래 6가지 유형 중 하나로 분류하고, 반드시 JSON만 출력하세요.

[유형 목록]
- factual       : 특정 수치·날짜·제한 조건 등 단일 사실 조회
- comparative   : 두 개 이상 사업·기관·조건의 비교
- summarization : 사업 전반 내용 요약·개요 설명 요청
- procedural    : 제출 방법·참여 절차·단계별 안내
- analytical    : 적합성 평가·장단점·전략적 분석
- filter_based  : 기관명·예산 범위·날짜 조건이 명시된 검색

출력 형식 (JSON만, 다른 텍스트 금지):
{"type": "<유형>", "reason": "<한 문장 이유>"}

사용자 질의: {query}"""


def _classify_llm(query: str, provider: str, model: str) -> tuple[QueryType, str]:
    """
    LLM을 이용한 정교한 질의 유형 분류
    """
    try:
        from src.generation.gen import generate_pure_text

        prompt = _LLM_CLASSIFY_PROMPT.format(query=query)
        raw = generate_pure_text(prompt, provider=provider, llm_model_name=model)

        raw_clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(raw_clean)

        type_str = parsed.get("type", "").strip().lower()
        reason   = parsed.get("reason", "LLM 분류")

        try:
            return QueryType(type_str), reason
        except ValueError:
            logger.warning(f"LLM이 알 수 없는 유형을 반환했습니다: '{type_str}'")
            return QueryType.UNKNOWN, "LLM 유형 미인식"

    except Exception as e:
        logger.warning(f"LLM 분류기 오류 (규칙 기반으로 fallback): {e}")
        return QueryType.UNKNOWN, "LLM 오류"


_DEFAULT_ROUTE_TABLE: dict[QueryType, tuple[str, int, float, PromptMode, bool]] = {
    QueryType.FACTUAL: (
        "keyword",  3,  0.0,  PromptMode.BASIC,    False
    ),
    QueryType.COMPARATIVE: (
        "hybrid",   5,  0.5,  PromptMode.COT_FEW,  True
    ),
    QueryType.SUMMARIZATION: (
        "vector",   4,  0.55, PromptMode.BASIC,    False
    ),
    QueryType.PROCEDURAL: (
        "hyde",     3,  0.55, PromptMode.COT_ZERO, False
    ),
    QueryType.ANALYTICAL: (
        "mmr",      5,  0.5,  PromptMode.COT_ZERO, True
    ),
    QueryType.FILTER_BASED: (
        "keyword",  5,  0.0,  PromptMode.BASIC,    False
    ),
    QueryType.UNKNOWN: (
        "hybrid",   3,  0.6,  PromptMode.BASIC,    False
    ),
}


def _build_route(
    query_type: QueryType,
    reason: str,
    default_provider: Optional[str] = None,
    default_model:    Optional[str] = None,
    router_cfg: Optional[dict] = None,
) -> RouteConfig:
    """
    질의 유형 → RouteConfig 변환. router_cfg로 개별 유형 override 가능
    """

    search_mode, top_k, threshold, prompt_mode, multi = _DEFAULT_ROUTE_TABLE[query_type]

    # config.yaml의 router.overrides.<type> 섹션으로 세부 재정의
    if router_cfg:
        overrides: dict = router_cfg.get("overrides", {}).get(query_type.value, {})
        search_mode = overrides.get("search_mode",     search_mode)
        top_k       = overrides.get("top_k",           top_k)
        threshold   = overrides.get("score_threshold", threshold)
        prompt_str  = overrides.get("prompt_mode",     prompt_mode.value)
        multi       = overrides.get("use_multi_query", multi)
        try:
            prompt_mode = PromptMode(prompt_str)
        except ValueError:
            pass

        # 유형별 LLM 재지정
        default_provider = overrides.get("llm_provider", default_provider)
        default_model    = overrides.get("llm_model",    default_model)

    return RouteConfig(
        query_type      = query_type,
        search_mode     = search_mode,
        top_k           = top_k,
        score_threshold = threshold,
        prompt_mode     = prompt_mode,
        llm_provider    = default_provider,
        llm_model       = default_model,
        use_multi_query = multi,
        reason          = reason,
    )


def route(
    query: str,
    *,
    use_llm_classifier:  bool         = False,
    llm_provider:        str          = "local",
    llm_model:           str          = "exaone3.5:7.8b",
    router_cfg:          Optional[dict] = None,
    force_query_type:    Optional[str] = None,
) -> RouteConfig:
    """
    질의를 분석하여 최적의 RouteConfig 반환
    """
    # 명시적 override
    if force_query_type:
        try:
            qtype = QueryType(force_query_type)
            cfg = _build_route(qtype, f"강제 지정: {force_query_type}",
                                llm_provider, llm_model, router_cfg)
            logger.info(cfg.describe())
            return cfg
        except ValueError:
            logger.warning(f"force_query_type 값이 유효하지 않습니다: '{force_query_type}'")

    # LLM 분류기
    if use_llm_classifier:
        qtype, reason = _classify_llm(query, llm_provider, llm_model)
        if qtype != QueryType.UNKNOWN:
            cfg = _build_route(qtype, f"LLM 분류 → {reason}",
                                llm_provider, llm_model, router_cfg)
            logger.info(cfg.describe())
            return cfg
        # LLM이 UNKNOWN 반환 → 규칙 기반으로 재시도
        logger.debug("LLM 분류 실패, 규칙 기반으로 재시도합니다.")

    # 규칙 기반 분류기
    qtype, reason = _classify_rule_based(query)
    cfg = _build_route(qtype, f"규칙 기반 → {reason}",
                        llm_provider, llm_model, router_cfg)
    logger.info(cfg.describe())
    return cfg