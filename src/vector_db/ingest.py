import uuid
from qdrant_client.models import PointStruct
from src.vector_db.vectordb import client
from src.embeddings.embedding import embed_text
from src.embeddings.sparse_embed import embed_sparse_text
from src.parsing.meta_db import normalize_source_filename

def ingest(embed_provider: str, collection_name: str, rag_data: list):
    print(f"총 {len(rag_data)}개의 데이터 벡터 DB 적재 시작...")
    
    def generate_points():
        for item in rag_data:
            m = dict(item.get("metadata", {}))
            if m.get("file_name"):
                m["file_name"] = normalize_source_filename(m["file_name"])

            text_to_embed = f"""
            제목: {m.get('title', '')}
            기관: {m.get('organization', '')}
            예산: {m.get('budget', 0)}
            공고일: {m.get('announcement_date', '')}
            입찰 시작: {m.get('bid_start', '')}
            입찰 마감: {m.get('bid_deadline', '')}
            섹션: {m.get('section_title', '')}
            내용: {m.get('content', '')}
            """
            
            dense_vector = embed_text(text_to_embed, provider=embed_provider)
            sparse_vector = embed_sparse_text(text_to_embed)
            
            # 고유 ID 생성
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item["id"]))

            yield PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector,
                },
                payload=m
            )
    
    client.upload_points(
        collection_name=collection_name,
        points=generate_points(),
        batch_size=100, # Qdrant의 내장 배치 업로드 기능 사용
        parallel=2 # 병렬 처리 워커 수
    )

    print(f"ingestion 완료 (컬렉션: {collection_name}, 모델: {embed_provider})")
