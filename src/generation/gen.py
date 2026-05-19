import requests
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def generate_answer(query, docs, provider="local", llm_model_name="exaone3.5:7.8b"):
    #파일명을 답변에 사용할 수 있도록 메타데이터의 file_name 을 context 맨위로 추가
    context_list = []
    for i, d in enumerate(docs):
        # file_name이 없을 경우를 대비해 기본값 설정
        file_name = d.get('file_name', '파일명 정보 없음')
        
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
    당신은 RFP/입찰 분석 전문가이다.

    답변은 반드시 한국어로 작성하고, 아래 규칙을 엄격히 준수하라:
    1. 정보를 찾았다면 해당 정보가 포함된 문서의 [출처 파일명]을 반드시 언급하며 답변하라.
    2. 답변의 마지막에 "출처 상세"를 정리해서 보여라.
    3. 만약 제공된 모든 문서에서 질문과 관련된 내용을 찾을 수 없다면, 다른 파일에 대해 구구절절 설명하지 말고 해당 질문에 관련된 정보를 찾을 수 없다 말해라. 만약 찾지 못했다면 2번 출처 상세를 표시하지 않아도 된다.
    4. 정보를 찾은 문서에 대해서만 이야기하고 다른 추측은 하지 않는다.

    출처 상세 형식:
    - 파일명: [정확한 파일명 기록]
    - 제목 / 기관 / 공고일 / 입찰기간
    - 핵심내용: (요약)

    문서:
    {context}

    질문:
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
        return answer, usage

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": llm_model_name,
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()
    answer = result["response"]
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
