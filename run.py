from rag.vectordb import create_collection
from rag.ingest import ingest
from rag.pipeline import rag_pipeline
from rag.config import EMBEDDING_PROVIDER

print("main 시작")

create_collection(provider=EMBEDDING_PROVIDER)

ingest()

query = "AWS 리소스가 필요한 사업 찾아줘"

result = rag_pipeline(query)

print("\n===== 답변 =====\n")

print(result)
