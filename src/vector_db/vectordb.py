from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from src.core.config import COLLECTION_MAP

client = QdrantClient(":memory:")

def create_collection(provider="local"):

    size_map = {
        "local": 1024,
        "openai": 1536
    }

    client.recreate_collection(
        collection_name=COLLECTION_MAP[provider],
        vectors_config=VectorParams(
            size=size_map[provider],
            distance=Distance.COSINE
        )
    )