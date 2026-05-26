import requests
from openai import OpenAI
import os
import re
from dotenv import load_dotenv
from src.parsing.meta_db import resolve_source_filename

load_dotenv()

_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _build_context(docs: list) -> str:
    context_list = []
    for d in docs:
        file_name = resolve_source_filename(d.get('file_name', '')) or '파일명 정보 없음'
        page_number = d.get('page_number')
        if page_number is None:
            page_number = '페이지 정보 없음'
        chunk_id = d.get('chunk_id') or d.get('id') or '청크 ID 정보 없음'

        doc_str = f"""
        [출처 파일: {file_name}]
        [페이지: {page_number}]
        [청크 ID: {chunk_id}]
        제목: {d.get('title','')}
        기관: {d.get('organization','')}
        예산: {d.get('budget','')}
        공고일: {d.get('announcement_date','')}
        입찰기간: {d.get('bid_start','')} ~ {d.get('bid_deadline','')}
        섹션: {d.get('section_title','')}
        내용: {d.get('content','')}
        """
        context_list.append(doc_str)
    return "\n\n".join(context_list)


def _source_entries(docs):
    source_entries = []
    seen = set()

    for doc in docs:
        file_name = resolve_source_filename(doc.get("file_name", ""))
        if not file_name:
            continue

        chunk_id = doc.get("chunk_id") or doc.get("id")
        page_number = doc.get("page_number")
        section_title = doc.get("section_title")
        key = chunk_id or (file_name, page_number, section_title)

        if key in seen:
            continue

        seen.add(key)
        source_entries.append({
            "file_name": file_name,
            "page_number": page_number,
            "chunk_id": chunk_id,
            "section_title": section_title,
        })

    return source_entries


def _format_source_entry(entry):
    parts = [entry["file_name"]]

    if entry.get("page_number") is not None:
        parts.append(f"p.{entry['page_number']}")

    if entry.get("chunk_id"):
        parts.append(f"chunk={entry['chunk_id']}")

    return f"출처: {', '.join(str(part) for part in parts if part)}"


def _strip_llm_source_section(answer):
    pattern = r"\n*(?:\*{0,2}\s*)?\[?\s*출처\s*(?:상세|파일)\s*\]?\s*(?:\*{0,2})?\s*\n.*$"
    return re.sub(pattern, "", answer.strip(), flags=re.DOTALL)


def _append_source_files(answer, docs):
    source_entries = _source_entries(docs)
    if not source_entries:
        return answer

    source_block = "\n".join(
        f"- {_format_source_entry(entry)}"
        for entry in source_entries
    )
    return f"{_strip_llm_source_section(answer)}\n\n**출처 상세**\n{source_block}"


def _build_user_prompt(query: str, context: str) -> str:
    """현재 질문과 context로 단일 user 메시지를 구성합니다."""
    return f"""당신은 RFP 분석 전문가입니다. 아래 [문서]를 근거로 [질문]에 답변하세요.

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


def generate_answer(
    query: str,
    docs: list,
    provider: str = "local",
    llm_model_name: str = "exaone3.5:7.8b",
    conversation_history: list = None,
):
    if conversation_history is None:
        conversation_history = []

    context = _build_context(docs)
    user_prompt = _build_user_prompt(query, context)

    # 공통 메시지 구성: system → 히스토리 → 현재 질문
    system_message = {"role": "system", "content": "RFP 분석 전문가"}
    messages = [system_message] + conversation_history + [{"role": "user", "content": user_prompt}]

    # OpenAI
    if provider == "openai":
        client = get_client()
        res = client.chat.completions.create(
            model=llm_model_name,
            messages=messages,
        )
        answer = res.choices[0].message.content
        usage = {
            "input": res.usage.prompt_tokens,
            "output": res.usage.completion_tokens,
            "total": res.usage.total_tokens,
        }
        return _append_source_files(answer, docs), usage

    # Local
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": llm_model_name,
            "messages": messages,
            "stream": False,
        },
    )
    result = response.json()
    answer = result["message"]["content"]
    usage = {
        "input": result.get("prompt_eval_count", 0),
        "output": result.get("eval_count", 0),
        "total": result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
    }
    return _append_source_files(answer, docs), usage


# HyDE 및 contextual retrieval 용 LLM 모듈
def generate_pure_text(prompt: str, provider: str = "local") -> str:
    if provider == "openai":
        client = get_client()
        res = client.chat.completions.create(
            model="gpt-5-nano",  # 현재 설정된 모델명 유지
            messages=[
                {"role": "system", "content": "RFP/공문서 작성 전문가"},
                {"role": "user", "content": prompt}
            ]
        )
        return res.choices[0].message.content

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "exaone3.5:7.8b",
                "prompt": prompt,
                "stream": False
            }
        )
        result = response.json()
        return result["response"]
    except Exception as e:
        print(f"[오류] 로컬 LLM 호출 실패: {e}")
        raise e
