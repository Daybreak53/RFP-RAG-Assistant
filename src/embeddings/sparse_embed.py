import hashlib
import logging
from collections import Counter, defaultdict
from functools import lru_cache

from kiwipiepy import Kiwi
from qdrant_client import models

# 로거 설정
logger = logging.getLogger(__name__)

TARGET_POS_TAGS = ('N', 'V', 'R', 'SL')   # 형태소 분석에서 추출할 타겟 품사 태그 (명사, 동사, 어근 등)
MIN_TOKEN_LENGTH = 2                      # 검색어로서 의미를 가지는 최소 글자 수


@lru_cache(maxsize=1)
def get_kiwi_instance() -> Kiwi:
    """
    Kiwi 형태소 분석기 인스턴스 로드
    """
    logger.info("Kiwi 형태소 분석기 인스턴스를 초기화합니다.")
    return Kiwi()


def get_stable_hash(word: str) -> int:
    """
    단어를 고정된 크기의 정수 인덱스(32비트)로 해싱
    """
    hash_bytes = hashlib.md5(word.encode('utf-8')).digest()
    return int.from_bytes(hash_bytes[:4], byteorder='big')


def embed_sparse_text(text: str) -> models.SparseVector:
    """
    텍스트를 형태소 분석하여 Sparse 벡터로 임베딩
    """
    if not text or not text.strip():
        logger.warning("빈 텍스트가 sparse 임베딩 함수로 전달되었습니다.")
        return models.SparseVector(indices=[], values=[])

    kiwi = get_kiwi_instance()

    try:
        # 형태소 분석 및 필터링 (의미 있는 품사이면서 길이가 MIN_TOKEN_LENGTH 이상인 토큰)
        tokens = [
            token.form for token in kiwi.tokenize(text)
            if token.tag.startswith(TARGET_POS_TAGS) and len(token.form) >= MIN_TOKEN_LENGTH
        ]
        
        # 추출된 토큰의 빈도수 계산
        token_counts = Counter(tokens)
        
        # 해시 인덱스 매핑 및 빈도수 누적
        sparse_dict = defaultdict(float)
        for token, count in token_counts.items():
            idx = get_stable_hash(token)
            sparse_dict[idx] += count
            
        # Qdrant 모델 구조에 맞게 반환
        return models.SparseVector(
            indices=list(sparse_dict.keys()),
            values=list(sparse_dict.values())
        )

    except Exception as e:
        logger.error(f"Sparse 임베딩 생성 중 형태소 분석 오류 발생: {e}")
        raise