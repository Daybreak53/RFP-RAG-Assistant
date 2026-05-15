from qdrant_client.models import PointStruct
from src.vector_db.vectordb import client
from src.embeddings.embedding import embed_text
from src.parsing.run_parsing import RAG_JSON_PATH
import json


def ingest(embed_provider: str, collection_name: str, file_path=RAG_JSON_PATH):
    print(f"데이터 로드 시작: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        return
    
    points = []

    for i, item in enumerate(data_list):
        m = item.get("metadata", {})
        
        text = f"""
        제목: {m.get('title', '')}
        기관: {m.get('organization', '')}
        예산: {m.get('budget', 0)}
        공고일: {m.get('announcement_date', '')}
        입찰 시작: {m.get('bid_start', '')}
        입찰 마감: {m.get('bid_deadline', '')}
        섹션: {m.get('section_title', '')}
        내용: {m.get('content', '')}
        """

        vector = embed_text(text, provider=embed_provider)

        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload=m #메타데이터에 있는거 어차피 다 쓰니까 코드간결화를 위해 통쨰로 넣음
            )
        )

        if len(points) >= 100: #api 사용을 위해 너무 과도한 양의 데이터가 전송되지 않도록 조절
            client.upsert(collection_name, points=points)
            points = []
 
    if points:       # 위에서 처리 안된 100개 미만의 벡터 전송
        client.upsert(collection_name, points=points)

    print(f"ingestion 완료 (컬렉션: {collection_name}, 모델: {embed_provider})")