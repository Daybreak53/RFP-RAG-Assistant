import logging
import uuid
from typing import Any, Dict, List, Iterator

from qdrant_client.models import PointStruct

from src.vector_db.vectordb import client
from src.embeddings.embedding import embed_text
from src.embeddings.sparse_embed import embed_sparse_text

# 로거 설정
logger = logging.getLogger(__name__)


def _build_embed_text(metadata: Dict[str, Any]) -> str:
    """
    메타데이터를 바탕으로 임베딩(Dense/Sparse) 생성을 위한 하나의 텍스트로 결합
    """
    return f"""
    제목: {metadata.get('title', '')}
    기관: {metadata.get('organization', '')}
    예산: {metadata.get('budget', 0)}
    공고일: {metadata.get('announcement_date', '')}
    입찰 시작: {metadata.get('bid_start', '')}
    입찰 마감: {metadata.get('bid_deadline', '')}
    섹션: {metadata.get('section_title', '')}
    내용: {metadata.get('content', '')}
    """.strip()


def ingest(
    embed_provider: str, 
    collection_name: str, 
    rag_data: List[Dict[str, Any]]
) -> None:
    """
    청킹 및 포맷팅된 RAG 데이터를 벡터 DB(Qdrant)에 적재
    """
    if not rag_data:
        logger.warning("적재할 데이터(rag_data)가 비어있어 Ingest를 건너뜁니다.")
        return

    logger.info(f"총 {len(rag_data)}개의 데이터 벡터 DB 적재 시작 (컬렉션: {collection_name})")

    def _embed_and_build_points(batch_items: List[Dict[str, Any]], provider: str) -> Iterator[PointStruct]:
        texts_to_embed = [_build_embed_text(item.get("metadata", {})) for item in batch_items]
        
        try:
            dense_vectors = embed_text(texts_to_embed, provider=provider)
            
            for item, text, dense_vec in zip(batch_items, texts_to_embed, dense_vectors):
                try:
                    sparse_vec = embed_sparse_text(text)
                    
                    raw_id = item.get("id", str(uuid.uuid4()))
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))

                    yield PointStruct(
                        id=point_id,
                        vector={
                            "dense": dense_vec,
                            "sparse": sparse_vec,
                        },
                        payload=item.get("metadata", {})
                    )
                except Exception as e:
                    chunk_id = item.get('id', 'Unknown')
                    logger.error(f"데이터 포인트 생성 중 오류 발생 (ID: {chunk_id}): {e}")
                    continue
        except Exception as e:
            logger.error(f"배치 임베딩 처리 중 오류 발생: {e}")


    def generate_points() -> Iterator[PointStruct]:
        """
        데이터를 BATCH_SIZE 만큼 묶어서 _embed_and_build_points로 전달하는 제너레이터
        """
        BATCH_SIZE = 64
        items_batch = []
        
        for item in rag_data:
            items_batch.append(item)
            if len(items_batch) == BATCH_SIZE:
                yield from _embed_and_build_points(items_batch, embed_provider)
                items_batch = []
                
        if items_batch:
            yield from _embed_and_build_points(items_batch, embed_provider)
    
    try:
        # Qdrant 내장 배치 업로드
        client.upload_points(
            collection_name=collection_name,
            points=generate_points(),
            batch_size=100,  # 한 번에 전송할 배치 크기
            parallel=2       # 병렬 처리 워커 수
        )
        logger.info(f"벡터 DB 적재(Ingestion) 완료 (컬렉션: {collection_name}, 모델: {embed_provider})")
        
    except Exception as e:
        logger.error(f"벡터 DB 데이터 적재 중 오류 발생: {e}", exc_info=True)
        raise