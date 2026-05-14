from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer("BAAI/bge-m3")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_text(text: str, provider="local"):
    if provider == "openai":
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    # default local
    return model.encode(text).tolist()