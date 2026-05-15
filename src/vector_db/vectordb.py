from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
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
        vectors_config=VectorParams(
            size=size_map[embed_provider],
            distance=Distance.COSINE    
        )
    )