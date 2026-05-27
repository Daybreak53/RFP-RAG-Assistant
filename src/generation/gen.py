from __future__ import annotations

import os
import re
import logging
import requests
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.parsing.meta_db import resolve_source_filename

load_dotenv()

logger = logging.getLogger(__name__)

_OPENAI_CLIENT: Optional[OpenAI] = None

DEFAULT_LOCAL_MODEL  = "exaone3.5:7.8b"
DEFAULT_OPENAI_MODEL = "gpt-5-nano"

LOCAL_CHAT_URL     = "http://localhost:11434/api/chat"
LOCAL_GENERATE_URL = "http://localhost:11434/api/generate"


def get_openai_client() -> OpenAI:
    """
    OpenAI 클라이언트 로드
    """
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            raise ValueError("OPENAI_API_KEY 환경 변수가 누락되었습니다. .env 파일을 확인하세요.")
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def _build_context(docs: list[dict[str, Any]]) -> str:
    """
    검색된 문서 리스트를 LLM 프롬프트에 주입할 문자열로 변환
    """
    if not docs:
        return "관련 문서를 찾을 수 없습니다."

    context_list = []
    for d in docs:
        file_name   = resolve_source_filename(d.get("file_name", "")) or "파일명 정보 없음"
        page_number = d.get("page_number")
        if page_number is None:
            page_number = "페이지 정보 없음"
        chunk_id    = d.get("chunk_id") or d.get("id") or "청크 ID 정보 없음"

        doc_str = (
            f"[출처 파일: {file_name}]\n"
            f"[페이지: {page_number}]\n"
            f"[청크 ID: {chunk_id}]\n"
            f"- 제목: {d.get('title', 'N/A')}\n"
            f"- 기관: {d.get('organization', 'N/A')}\n"
            f"- 예산: {d.get('budget', 'N/A')}\n"
            f"- 공고일: {d.get('announcement_date', 'N/A')}\n"
            f"- 입찰기간: {d.get('bid_start', 'N/A')} ~ {d.get('bid_deadline', 'N/A')}\n"
            f"- 섹션: {d.get('section_title', 'N/A')}\n"
            f"- 내용: {d.get('content', '내용 없음')}"
        )
        context_list.append(doc_str)

    return "\n\n".join(context_list)


def _source_entries(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_entries = []
    seen = set()

    for doc in docs:
        file_name = resolve_source_filename(doc.get("file_name", ""))
        if not file_name:
            continue

        chunk_id      = doc.get("chunk_id") or doc.get("id")
        page_number   = doc.get("page_number")
        section_title = doc.get("section_title")
        key           = chunk_id or (file_name, page_number, section_title)

        if key in seen:
            continue

        seen.add(key)
        source_entries.append({
            "file_name":     file_name,
            "page_number":   page_number,
            "chunk_id":      chunk_id,
            "section_title": section_title,
        })

    return source_entries


def _format_source_entry(entry: dict[str, Any]) -> str:
    parts = [entry["file_name"]]

    if entry.get("page_number") is not None:
        parts.append(f"p.{entry['page_number']}")

    if entry.get("chunk_id"):
        parts.append(f"chunk={entry['chunk_id']}")

    return f"출처: {', '.join(str(part) for part in parts if part)}"


def _strip_llm_source_section(answer: str) -> str:
    pattern = r"\n*(?:\*{0,2}\s*)?\[?\s*출처\s*(?:상세|파일)\s*\]?\s*(?:\*{0,2})?\s*\n.*$"
    return re.sub(pattern, "", answer.strip(), flags=re.DOTALL)


def _append_source_files(answer: str, docs: list[dict[str, Any]]) -> str:
    source_entries = _source_entries(docs)
    if not source_entries:
        return answer

    source_block = "\n".join(
        f"- {_format_source_entry(entry)}"
        for entry in source_entries
    )
    return f"{_strip_llm_source_section(answer)}\n\n**출처 상세**\n{source_block}"


# 시스템 메시지  ―  질의 유형(QueryType)별 특화
_SYSTEM_MESSAGES: dict[str, str] = {
    "factual": (
        "당신은 꼼꼼하고 정확한 RFP 분석 전문가입니다. "
        "문서에 명시된 수치·날짜·제한 조건을 정확하게 인용하고, "
        "추측이나 근사치를 사용하지 마세요."
    ),
    "comparative": (
        "당신은 여러 공공입찰 RFP 문서를 동시에 비교·분석하는 전문가입니다. "
        "문서 간 차이점을 명확히 구분하고, 혼용하지 마세요."
    ),
    "summarization": (
        "당신은 RFP 문서의 핵심 내용을 간결하고 구조적으로 요약하는 전문가입니다. "
        "사업 목적, 주요 조건, 일정 순으로 정리하세요."
    ),
    "procedural": (
        "당신은 공공입찰 참여 절차와 제안서 작성 방법을 안내하는 전문가입니다. "
        "단계별로 명확하게 설명하고, 누락된 단계가 없도록 하세요."
    ),
    "analytical": (
        "당신은 공공입찰 전략과 RFP 요건을 심층 분석하는 전문가입니다. "
        "논리적 근거를 명시하고, 주관적 판단을 내릴 때는 반드시 근거를 제시하세요."
    ),
    "filter_based": (
        "당신은 특정 조건(기관·예산·기간)으로 필터링된 RFP 문서를 검토하는 전문가입니다. "
        "조건에 맞는 문서만 참조하고, 조건 외 문서는 언급하지 마세요."
    ),
    "unknown": (
        "당신은 꼼꼼하고 정확한 RFP 분석 전문가입니다."
    ),
    "default": (
        "당신은 꼼꼼하고 정확한 RFP 분석 전문가입니다."
    ),
}


def _build_prompt_basic(query: str, context: str, query_type: str) -> str:
    """
    기본 4규칙 + 출처 상세 형식 프롬프트
    """
    return f"""아래 [문서]를 근거로 [질문]에 답변하세요.

[규칙]
1. 근거 기반: [문서]에 없는 내용은 절대 답변하지 마세요(정보 부족 시 "정보를 찾을 수 없습니다" 출력).
2. 출처 명시: 답변 내용에 [파일명]을 반드시 표기하세요.
3. 문서 구분: 여러 문서의 정보가 상이하면 절대 섞지 말고, 문서별로 나누어 서술하세요.
4. 출력 형식: 간략한 '사고 과정' 후 '답변:'을 제시하고, 마지막에 [출처 상세]를 작성하세요.
5. 이전 대화가 있다면, 대화 맥락을 참고하여 답변하세요.

[출처 상세 형식]
- 파일명: [파일명]
- [제목] / [기관] / [공고일] / [입찰기간]
- 핵심내용: (1줄 요약)

[문서]
{context}

[질문]
{query}"""


def _build_prompt_cot_zero(query: str, context: str, query_type: str) -> str:
    """
    Zero-shot CoT 프롬프트
    - 질의 유형에 맞는 추론 지침을 자동 삽입
    - "단계적으로 생각하세요" 지시로 논리 오류 억제
    """
    # 유형별 추가 지침
    type_hints: dict[str, str] = {
        "factual": (
            "먼저 문서에서 해당 수치나 날짜를 찾은 뒤, "
            "단위와 조건을 정확히 확인하세요."
        ),
        "procedural": (
            "먼저 전체 절차를 파악한 뒤, "
            "각 단계를 번호를 붙여 순서대로 서술하세요."
        ),
        "analytical": (
            "먼저 분석 대상을 명확히 정의한 뒤, "
            "장점-단점-결론 구조로 논리적으로 서술하세요."
        ),
        "comparative": (
            "먼저 비교 대상 문서를 각각 파악한 뒤, "
            "공통점과 차이점을 표 또는 목록으로 정리하세요."
        ),
        "summarization": (
            "먼저 핵심 키워드를 추출한 뒤, "
            "사업 목적 → 주요 조건 → 일정 순으로 요약하세요."
        ),
    }
    hint = type_hints.get(query_type, "문서를 꼼꼼히 읽고 단계적으로 생각하세요.")

    return f"""아래 [문서]를 근거로 [질문]에 답변하세요.

[사고 지침]
- {hint}
- 단계적으로 생각(Think Step by Step)한 후 최종 답변을 작성하세요.
- 추론 과정을 '▶ 사고:' 섹션에 기록하고, 결론을 '▶ 답변:' 섹션에 작성하세요.
- [문서]에 없는 내용은 절대 작성하지 마세요.
- 출처 파일명을 답변 내에 반드시 표기하세요.

[출력 형식]
▶ 사고:
(추론 과정을 단계별로 서술)

▶ 답변:
(최종 답변)

[출처]
- 파일명: [파일명]
- 핵심내용: (1줄 요약)

[문서]
{context}

[질문]
{query}"""


_FEW_SHOT_EXAMPLES: dict[str, str] = {
    "factual": """
[예시]
질문: 제안서 본문의 최대 페이지 수는?
▶ 사고:
  Step1. "페이지", "최대", "제한"을 키워드로 문서 탐색.
  Step2. 문서에서 "제안서 본문은 00페이지 이내로 작성" 문구 발견.
  Step3. 수치와 단위를 그대로 인용.
▶ 답변:
  [OO제안요청서.hwp] 기준, 제안서 본문의 최대 페이지 수는 **00페이지**입니다.
[출처] 파일명: OO제안요청서.hwp / 핵심: 제안서 분량 제한 규정""",

    "comparative": """
[예시]
질문: A 사업과 B 사업의 입찰 마감일 차이는?
▶ 사고:
  Step1. A 문서에서 입찰 마감일 확인.
  Step2. B 문서에서 입찰 마감일 확인.
  Step3. 두 날짜의 차이를 계산.
▶ 답변:
  - [A사업_RFP.pdf]: 입찰 마감일 YYYY-MM-DD
  - [B사업_RFP.pdf]: 입찰 마감일 YYYY-MM-DD
  → 두 사업의 마감일 차이는 OO일입니다.
[출처] A사업_RFP.pdf, B사업_RFP.pdf""",

    "summarization": """
[예시]
질문: 이 RFP 사업의 주요 내용을 요약해줘.
▶ 사고:
  Step1. 사업명, 발주기관, 목적 파악.
  Step2. 주요 요구사항 및 조건 정리.
  Step3. 예산 및 일정 확인.
▶ 답변:
  [OO구축사업_RFP.pdf] 요약:
  • 사업 목적: ...
  • 주요 조건: ...
  • 예산·일정: ...
[출처] OO구축사업_RFP.pdf""",

    "procedural": """
[예시]
질문: 제안서 제출 절차는?
▶ 사고:
  Step1. 문서에서 "제출", "접수", "방법" 관련 항목 탐색.
  Step2. 절차를 순서대로 나열.
▶ 답변:
  [OO제안요청서.pdf] 기준 제안서 제출 절차:
  1단계. 사전 등록: ...
  2단계. 서류 준비: ...
  3단계. 제출 방법: ...
[출처] OO제안요청서.pdf""",

    "analytical": """
[예시]
질문: 이 사업의 평가 배점 구조가 적합한가?
▶ 사고:
  Step1. 평가 항목 및 배점 파악.
  Step2. 일반적인 공공입찰 기준과 비교.
  Step3. 장단점 분석 후 종합 판단.
▶ 답변:
  [OO평가기준.pdf] 분석 결과:
  • 장점: ...
  • 단점: ...
  • 종합 의견: ...
[출처] OO평가기준.pdf""",
}


def _build_prompt_cot_few(query: str, context: str, query_type: str) -> str:
    """
    Few-shot CoT 프롬프트. 질의 유형별 예시를 동적으로 삽입
    """
    example = _FEW_SHOT_EXAMPLES.get(
        query_type,
        _FEW_SHOT_EXAMPLES.get("factual", "")
    )

    return f"""아래 [문서]를 근거로 [질문]에 답변하세요.
[예시]를 참고하여 같은 형식으로 작성하되, 예시 내용을 그대로 복사하지 마세요.

[규칙]
- [문서]에 없는 내용은 절대 작성하지 마세요.
- 출처 파일명을 답변 내에 반드시 표기하세요.
- 이전 대화가 있다면 맥락을 반영하세요.

{example}

[문서]
{context}

[질문]
{query}"""


def build_user_prompt(
    query: str,
    context: str,
    prompt_mode: str = "basic",
    query_type: str  = "unknown",
) -> str:
    """
    prompt_mode와 query_type 조합으로 최적 유저 프롬프트 반환
    """
    mode = prompt_mode.lower()

    if mode == "cot_zero":
        return _build_prompt_cot_zero(query, context, query_type)
    elif mode == "cot_few":
        return _build_prompt_cot_few(query, context, query_type)
    else:
        return _build_prompt_basic(query, context, query_type)


def generate_answer(
    query: str,
    docs: list[dict[str, Any]],
    provider: str = "local",
    llm_model_name: str = DEFAULT_LOCAL_MODEL,
    conversation_history: Optional[list[dict[str, str]]] = None,
    prompt_mode: str = "basic",
    query_type:  str = "unknown",
) -> tuple[str, dict[str, int]]:
    """
    RAG 파이프라인 최종 답변 생성
    """
    history = conversation_history or []
    context = _build_context(docs)

    # 질의 유형별 시스템 메시지 선택
    system_content = _SYSTEM_MESSAGES.get(query_type, _SYSTEM_MESSAGES["default"])

    # 프롬프트 모드별 유저 프롬프트 생성
    user_prompt = build_user_prompt(query, context, prompt_mode, query_type)

    system_message = {"role": "system", "content": system_content}
    messages = [system_message] + history + [{"role": "user", "content": user_prompt}]

    # 사용된 설정 로깅
    logger.info(
        f"[generate_answer] provider={provider} | model={llm_model_name} | "
        f"prompt_mode={prompt_mode} | query_type={query_type}"
    )

    try:
        if provider == "openai":
            client = get_openai_client()
            res = client.chat.completions.create(
                model=llm_model_name,
                messages=messages,
            )
            answer = res.choices[0].message.content or ""
            usage = {
                "input":  res.usage.prompt_tokens     if res.usage else 0,
                "output": res.usage.completion_tokens if res.usage else 0,
                "total":  res.usage.total_tokens      if res.usage else 0,
            }
            return _append_source_files(answer, docs), usage

        elif provider == "local":
            response = requests.post(
                LOCAL_CHAT_URL,
                json={"model": llm_model_name, "messages": messages, "stream": False},
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            answer = result.get("message", {}).get("content", "")
            p = result.get("prompt_eval_count", 0)
            e = result.get("eval_count", 0)
            usage = {"input": p, "output": e, "total": p + e}
            return _append_source_files(answer, docs), usage

        else:
            raise ValueError(f"지원하지 않는 LLM 제공자입니다: {provider}")

    except requests.RequestException as err:
        logger.error(f"로컬 LLM 통신 오류: {err}", exc_info=True)
        return "답변 생성 중 LLM 서버와의 통신 오류가 발생했습니다.", {"input": 0, "output": 0, "total": 0}
    except Exception as err:
        logger.error(f"답변 생성 중 시스템 오류: {err}", exc_info=True)
        return "답변 생성 중 시스템 오류가 발생했습니다.", {"input": 0, "output": 0, "total": 0}


def generate_pure_text(
    prompt: str,
    provider: str = "local",
    llm_model_name: Optional[str] = None,
) -> str:
    """
    단순 텍스트 생성 경량 함수
    HyDE 가상 문서 생성, Multi-Query 확장, Contextual 요약 등에 사용
    """
    try:
        if provider == "openai":
            model  = llm_model_name or DEFAULT_OPENAI_MODEL
            client = get_openai_client()
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "당신은 행정 문서 및 RFP/공문서 작성 전문가입니다."},
                    {"role": "user",   "content": prompt},
                ],
            )
            return res.choices[0].message.content or ""

        elif provider == "local":
            model = llm_model_name or DEFAULT_LOCAL_MODEL
            response = requests.post(
                LOCAL_GENERATE_URL,
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("response", "")

        else:
            raise ValueError(f"지원하지 않는 LLM 제공자입니다: {provider}")

    except Exception as err:
        logger.error(f"순수 텍스트 생성 오류 (provider={provider}): {err}", exc_info=True)
        raise RuntimeError(f"LLM 텍스트 생성에 실패했습니다: {err}") from err