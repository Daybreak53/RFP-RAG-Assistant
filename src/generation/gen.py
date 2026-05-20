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


def _unique_source_files(docs):
    source_files = []
    seen = set()

    for doc in docs:
        file_name = resolve_source_filename(doc.get("file_name", ""))
        if not file_name or file_name in seen:
            continue
        seen.add(file_name)
        source_files.append(file_name)

    return source_files


def _strip_llm_source_section(answer):
    pattern = r"\n*\*{0,2}출처\s*상세\*{0,2}\s*\n.*$"
    return re.sub(pattern, "", answer.strip(), flags=re.DOTALL)


def _append_source_files(answer, docs):
    source_files = _unique_source_files(docs)
    if not source_files:
        return answer

    source_block = "\n".join(f"- {file_name}" for file_name in source_files)
    return f"{_strip_llm_source_section(answer)}\n\n**출처 파일**\n{source_block}"

def generate_answer(query, docs, provider="local", llm_model_name="exaone3.5:7.8b"):
    #파일명을 답변에 사용할 수 있도록 메타데이터의 file_name 을 context 맨위로 추가
    context_list = []
    for i, d in enumerate(docs):
        # file_name이 없을 경우를 대비해 기본값 설정
        file_name = resolve_source_filename(d.get('file_name', '')) or '파일명 정보 없음'
        
        doc_str = f"""
        [출처 파일: {file_name}]
        제목: {d.get('title','')}
        기관: {d.get('organization','')}
        예산: {d.get('budget','')}
        공고일: {d.get('announcement_date','')}
        입찰기간: {d.get('bid_start','')} ~ {d.get('bid_deadline','')}
        섹션: {d.get('section_title','')}
        내용: {d.get('content','')}
        """
        context_list.append(doc_str)
    
    context = "\n\n".join(context_list)


    #좀더 깔끔하게 보이도록 프롬프트 수정
    prompt = f"""
    당신은 RFP 분석 전문가입니다. 아래 [문서]를 근거로 [질문]에 답변하세요.

    [규칙]
    1. 근거 기반: [문서]에 없는 내용은 절대 답변하지 마세요(정보 부족 시 "정보를 찾을 수 없습니다" 출력).
    2. 출처 명시: 답변 내용에 [파일명]을 반드시 표기하세요.
    3. 문서 구분: 여러 문서의 정보가 상이하면 절대 섞지 말고, 문서별로 나누어 서술하세요.
    4. 출력 형식: 간략한 '사고 과정' 후 '답변:'을 제시하고, 마지막에 [출처 상세]를 작성하세요.

    [출처 상세 형식]
    - 파일명: [파일명]
    - [제목] / [기관] / [공고일] / [입찰기간]
    - 핵심내용: (1줄 요약)

    [문서]
    {context}

    [질문]
    {query}
    """
    
    if provider == "openai":
        client = get_client()
        res = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": "RFP 분석 전문가"},
                {"role": "user", "content": prompt}
            ]
        )
        answer = res.choices[0].message.content
        usage = {
            "input": res.usage.prompt_tokens,
            "output": res.usage.completion_tokens,
            "total": res.usage.total_tokens,
        }
        return _append_source_files(answer, docs), usage

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": llm_model_name,
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()
    answer = _append_source_files(result["response"], docs)
    usage = {
        "input": result.get("prompt_eval_count", 0),
        "output": result.get("eval_count", 0),
        "total": result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
    }
    return answer, usage




# HyDE 및 다목적 텍스트 생성을 위한 헬퍼 함수

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
