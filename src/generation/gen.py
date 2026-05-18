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
            "model": "exaone3.5:7.8b",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()

    answer = result["response"]

    return answer

if __name__ == '__main__':
    print("EXAONE 3.5 7.8B 모델 테스트를 시작합니다...")
    
    # 1. 테스트용 가상 문서 데이터 (docs)
    sample_docs = [
        {
            'file_name': 'A공공기관_AI구축_RFP.pdf',
            'title': '2026년 인공지능 기반 행정 서비스 구축 사업',
            'organization': '행정안전부',
            'budget': '15억원',
            'announcement_date': '2026-05-01',
            'bid_start': '2026-05-02',
            'bid_deadline': '2026-06-01',
            'section_title': '제안 요구 사항',
            'content': '본 사업은 자체 거대언어모델(LLM)을 활용한 문서 요약 및 검색 증강 생성(RAG) 시스템 구현을 핵심 골자로 한다.'
        },
        {
            'file_name': 'B공공기관_보안 가이드라인.txt',
            'title': '정보화 사업 보안 대책 강화를 위한 지침',
            'organization': '국가정보원',
            'budget': '해당없음',
            'announcement_date': '2025-12-10',
            'bid_start': '해당없음',
            'bid_deadline': '해당없음',
            'section_title': '데이터 저장 보안',
            'content': '모든 학습 데이터 및 LLM API 로그는 외부 클라우드가 아닌 원격 사설 서버(On-Premise) 내에 독립적으로 저장되어야 한다.'
        }
    ]
    
    # 2. 테스트용 질문 (query)
    # 일부러 문서에 있는 '예산'과 '보안'을 물어보는 질문입니다.
    sample_query = "AI 구축 사업의 예산은 얼마이고, 데이터는 어디에 저장해야 하나요?"
    
    print(f"질문: {sample_query}\n")
    print("엑사원 엔진이 답변을 생성하는 중입니다... 잠시만 기다려주세요.")
    
    try:
        # 3. 우리가 만든 generate_answer 함수 호출 (provider="local"이 기본값)
        final_answer = generate_answer(query=sample_query, docs=sample_docs, provider="local")
        
        print("\n" + "="*50)
        print("[엑사원 7.8B 최종 답변]")
        print("="*50)
        print(final_answer)
        print("="*50)
        print("\n테스트가 성공적으로 완료되었습니다! API 통신 정상입니다.")
        
    except Exception as e:
        print("\n에러 발생! 테스트에 실패했습니다.")
        print(f"에러 내용: {e}")