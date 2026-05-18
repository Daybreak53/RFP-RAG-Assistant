from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

_model_instance = None
_openai_client = None

def get_dense_model():
    global _model_instance
    if _model_instance == None:
        # 처음 호출될 때 딱 한 번만 가중치를 로드합니다.
        _model_instance = SentenceTransformer("BAAI/bge-m3")
    return _model_instance

def get_openai_client():
    global _openai_client
    if _openai_client == None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client

def embed_text(text: str, provider="local"):
    if provider == "openai":
        client = get_openai_client()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    # 'local'일 경우에만 무거운 로컬 모델을 불러와 재사용합니다.
    model = get_dense_model()
    return model.encode(text).tolist()
