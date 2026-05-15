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

def generate_answer(query, docs, provider="local"):

    context = "\n\n".join([
        f"""
        [문서 {i+1}]
        제목: {d.get('title','')}
        기관: {d.get('organization','')}
        예산: {d.get('budget','')}
        공고일: {d.get('announcement_date','')}
        입찰기간: {d.get('bid_start','')} ~ {d.get('bid_deadline','')}
        섹션: {d.get('section_title','')}
        내용: {d.get('content','')}
        """
        for i, d in enumerate(docs)
    ])

    prompt = f"""
    당신은 RFP/입찰 분석 전문가이다.

    답변은 반드시 한국어로 작성하고,
    마지막에 반드시 "출처"를 정리해서 보여라.
    없으면 없다고 답변해라.
    정보를 찾았다면 찾은 문서에 대한 이야기만 하고 다른 문서에 대한 이야기는 하지 않는다.

    출처 형식:
    - 제목 / 기관 / 공고일 / 입찰기간 / 핵심내용

    문서:
    {context}

    질문:
    {query}
    """
    if provider == "openai":
        client = get_client()
        res = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "RFP 분석 전문가"},
                {"role": "user", "content": prompt}
            ]
        )
        return res.choices[0].message.content

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen3:8b",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()

    answer = result["response"]

    return answer