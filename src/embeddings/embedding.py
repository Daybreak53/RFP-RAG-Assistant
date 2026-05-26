import os
import logging
from functools import lru_cache
from typing import List, Literal

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 로거 설정
logger = logging.getLogger(__name__)

# 임베딩 모델 상수
DEFAULT_LOCAL_MODEL = "BAAI/bge-m3"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


@lru_cache(maxsize=1)
def get_dense_model() -> SentenceTransformer:
    """
    로컬 임베딩 모델 로드
    """
    logger.info(f"로컬 임베딩 모델 로드 중: {DEFAULT_LOCAL_MODEL}")
    return SentenceTransformer(DEFAULT_LOCAL_MODEL)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """
    OpenAI 클라이언트 초기화
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    
    return OpenAI(api_key=api_key)


def embed_text(
    texts: List[str], 
    provider: Literal["local", "openai"] = "local"
) -> List[float]:
    """
    텍스트를 입력받아 벡터 임베딩 반환
    """
    if not texts:
        logger.warning("빈 텍스트 리스트가 임베딩 함수로 전달되었습니다.")
        return []

    try:
        if provider == "openai":
            client = get_openai_client()
            response = client.embeddings.create(
                model=DEFAULT_OPENAI_MODEL,
                input=texts
            )
            return [d.embedding for d in response.data]
        
        elif provider == "local":
            model = get_dense_model()
            return model.encode(texts, batch_size=len(texts)).tolist()
        
        else:
            raise ValueError(f"지원하지 않는 임베딩 provider 입니다: '{provider}'")
            
    except Exception as e:
        logger.error(f"'{provider}' 임베딩 생성 중 오류 발생: {e}")
        raise