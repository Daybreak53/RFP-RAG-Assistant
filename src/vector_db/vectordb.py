import logging
import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# 환경 변수 로드
load_dotenv()

# 로거 설정
logger = logging.getLogger(__name__)

_QDRANT_URL = os.getenv("QDRANT_URL")
_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not _QDRANT_URL:
    logger.warning("QDRANT_URL 환경 변수가 설정되지 않았습니다. (로컬/인메모리 모드로 동작할 수 있습니다.)")

# 전역 클라이언트 인스턴스 (다른 모듈에서 import하여 사용)
client = QdrantClient(
    url=_QDRANT_URL,
    api_key=_QDRANT_API_KEY
)

# 메타데이터 필터에 사용할 payload 인덱스 정의
_PAYLOAD_INDEXES: List[Tuple[str, models.PayloadSchemaType]] = [
    ("organization",       models.PayloadSchemaType.TEXT),
    ("title",              models.PayloadSchemaType.TEXT),
    ("doc_id",             models.PayloadSchemaType.KEYWORD),
    ("file_type",          models.PayloadSchemaType.KEYWORD),
    ("budget",             models.PayloadSchemaType.FLOAT),
    ("announcement_date",  models.PayloadSchemaType.DATETIME),
    ("bid_start",          models.PayloadSchemaType.DATETIME),
    ("bid_deadline",       models.PayloadSchemaType.DATETIME),
]

# 임베딩 모델(Provider)별 벡터 차원 수
_VECTOR_SIZE_MAP: Dict[str, int] = {
    "local": 1024,   # BAAI/bge-m3 모델 기준
    "openai": 1536   # text-embedding-3-small 모델 기준
}


def create_collection(embed_provider: str, collection_name: str) -> None:
    """
    주어진 설정에 맞게 Qdrant 컬렉션을 생성하고, 메타데이터 필터용 인덱스 설정
    """
    if embed_provider not in _VECTOR_SIZE_MAP:
        raise ValueError(
            f"지원하지 않는 임베딩 제공자입니다: '{embed_provider}'. "
            f"지원 목록: {list(_VECTOR_SIZE_MAP.keys())}"
        )

    vector_size = _VECTOR_SIZE_MAP[embed_provider]
    logger.info(f"Qdrant 컬렉션 '{collection_name}' 생성을 시작합니다. (Dense 벡터 차원: {vector_size})")

    try:
        # 기존 컬렉션 존재 여부 확인 후 안전하게 삭제
        if client.collection_exists(collection_name=collection_name):
            logger.info(f"기존 컬렉션 '{collection_name}'을 발견하여 삭제합니다.")
            client.delete_collection(collection_name=collection_name)

        # 하이브리드 검색을 위한 Dense & Sparse 설정으로 컬렉션 생성
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE    
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF 
                )
            }
        )
        logger.info(f"컬렉션 '{collection_name}' (Dense+Sparse 하이브리드) 생성 완료")

        # 메타데이터 기반 필터링 속도 향상을 위한 Payload 인덱스 생성
        for field_name, schema_type in _PAYLOAD_INDEXES:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )

        logger.info(f"컬렉션 '{collection_name}'의 메타데이터 페이로드 인덱스(Payload Index) 설정 완료")

    except Exception as e:
        logger.error(f"컬렉션 '{collection_name}' 생성 및 설정 중 오류 발생: {e}", exc_info=True)
        raise