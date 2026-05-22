import os
import logging
import requests
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 로거 설정
logger = logging.getLogger(__name__)

_OPENAI_CLIENT: Optional[OpenAI] = None

DEFAULT_LOCAL_MODEL = "exaone3.5:7.8b"
DEFAULT_OPENAI_MODEL = "gpt-5-nano"

# 로컬 LLM(Ollama) API 엔드포인트
LOCAL_CHAT_URL = "http://localhost:11434/api/chat"
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
        file_name = d.get('file_name', '파일명 정보 없음')
        doc_str = (
            f"[출처 파일: {file_name}]\n"
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


def _build_user_prompt(query: str, context: str) -> str:
    """
    현재 질문과 검색된 context를 결합하여 사용자용 프롬프트를 구성합니다.
    """
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
    docs: list[dict[str, Any]],
    provider: str = "local",
    llm_model_name: str = DEFAULT_LOCAL_MODEL,
    conversation_history: Optional[list[dict[str, str]]] = None,
) -> tuple[str, dict[str, int]]:
    """
    RAG 파이프라인 최종 답변 생성
    """
    history = conversation_history or []
    context = _build_context(docs)
    user_prompt = _build_user_prompt(query, context)

    system_message = {"role": "system", "content": "당신은 꼼꼼하고 정확한 RFP 분석 전문가입니다."}
    messages = [system_message] + history + [{"role": "user", "content": user_prompt}]

    try:
        if provider == "openai":
            client = get_openai_client()
            res = client.chat.completions.create(
                model=llm_model_name,
                messages=messages,
                temperature=0.1,
            )
            answer = res.choices[0].message.content or ""
            usage = {
                "input": res.usage.prompt_tokens if res.usage else 0,
                "output": res.usage.completion_tokens if res.usage else 0,
                "total": res.usage.total_tokens if res.usage else 0,
            }
            return answer, usage

        elif provider == "local":
            response = requests.post(
                LOCAL_CHAT_URL,
                json={
                    "model": llm_model_name,
                    "messages": messages,
                    "stream": False,
                },
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("message", {}).get("content", "")
            
            prompt_eval_count = result.get("prompt_eval_count", 0)
            eval_count = result.get("eval_count", 0)
            usage = {
                "input": prompt_eval_count,
                "output": eval_count,
                "total": prompt_eval_count + eval_count,
            }
            return answer, usage

        else:
            raise ValueError(f"지원하지 않는 LLM 제공자입니다: {provider}")

    except requests.RequestException as e:
        logger.error(f"로컬 LLM({llm_model_name}) 통신 중 오류 발생: {e}", exc_info=True)
        return "답변 생성 중 LLM 서버와의 통신 오류가 발생했습니다.", {"input": 0, "output": 0, "total": 0}
    except Exception as e:
        logger.error(f"답변 생성 중 예기치 않은 시스템 오류 발생: {e}", exc_info=True)
        return "답변 생성 중 시스템 오류가 발생했습니다.", {"input": 0, "output": 0, "total": 0}


def generate_pure_text(
    prompt: str, 
    provider: str = "local",
    llm_model_name: Optional[str] = None
) -> str:
    """
    단순 텍스트 생성을 위한 경량 LLM 호출 함수
    주로 HyDE(가상 문서 생성) 및 Contextual Retrieval(맥락 요약) 파이프라인에서 사용
    """
    try:
        if provider == "openai":
            model = llm_model_name or DEFAULT_OPENAI_MODEL
            client = get_openai_client()
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "당신은 행정 문서 및 RFP/공문서 작성 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return res.choices[0].message.content or ""

        elif provider == "local":
            model = llm_model_name or DEFAULT_LOCAL_MODEL
            response = requests.post(
                LOCAL_GENERATE_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")

        else:
            raise ValueError(f"지원하지 않는 LLM 제공자입니다: {provider}")

    except Exception as e:
        logger.error(f"순수 텍스트 생성 중 오류 발생(provider={provider}): {e}", exc_info=True)
        raise RuntimeError(f"LLM 텍스트 생성에 실패했습니다: {e}") from e