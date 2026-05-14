from qdrant_client.models import PointStruct
from src.vector_db.vectordb import client
from src.embeddings.embedding import embed_text
from data.dummy_data import dummy_data


def ingest(embed_provider: str, collection_name: str):

    points = []

    for i, item in enumerate(dummy_data):

        m = item["metadata"]

        text = f"""
        제목: {m['title']}
        기관: {m['organization']}
        예산: {m['budget']}
        공고일: {m['announcement_date']}
        입찰 시작: {m['bid_start']}
        입찰 마감: {m['bid_deadline']}
        섹션: {m['section_title']}
        내용: {m['content']}
        """

        vector = embed_text(text, provider=embed_provider)

        points.append(
            PointStruct(
                id=i,           
                vector=vector,
                payload={
                    "doc_id": m["doc_id"],
                    "chunk_id": m["chunk_id"],
                    "title": m["title"],
                    "organization": m["organization"],
                    "budget": m["budget"],
                    "announcement_date": m["announcement_date"],
                    "bid_start": m["bid_start"],
                    "bid_deadline": m["bid_deadline"],
                    "page_number": m["page_number"],
                    "section_title": m["section_title"],
                    "content": m["content"],
                    "file_name": m["file_name"],
                    "file_type": m["file_type"],
                }
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )

    print(f"ingestion 완료 (컬렉션: {collection_name}, 모델: {embed_provider})")