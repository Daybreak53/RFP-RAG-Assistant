from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

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
    print(f"[벡터 DB] 컬렉션 '{collection_name}' 생성 및 텍스트 인덱스 설정 완료")