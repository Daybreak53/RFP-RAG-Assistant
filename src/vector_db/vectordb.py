from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# 메타데이터 필터에 사용할 payload 인덱스 정의
_PAYLOAD_INDEXES: list[tuple[str, models.PayloadSchemaType]] = [
    ("organization",       models.PayloadSchemaType.TEXT),
    ("title",              models.PayloadSchemaType.TEXT),
    ("doc_id",             models.PayloadSchemaType.KEYWORD),
    ("file_type",          models.PayloadSchemaType.KEYWORD),
    ("budget",             models.PayloadSchemaType.FLOAT),
    ("announcement_date",  models.PayloadSchemaType.KEYWORD),
    ("bid_start",          models.PayloadSchemaType.KEYWORD),
    ("bid_deadline",       models.PayloadSchemaType.KEYWORD),
]

def create_collection(embed_provider, collection_name):
    size_map = {
        "local": 1024,
        "openai": 1536
    }

    client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=size_map[embed_provider],
                distance=models.Distance.COSINE    
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF 
            )
        }
    )
    print(f"[벡터 DB] 컬렉션 '{collection_name}' 생성 완료")

    for field_name, schema_type in _PAYLOAD_INDEXES:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_type,
        )

    print(f"[벡터 DB] 텍스트 인덱스 설정 완료 (컬렉션: '{collection_name}')")