from dotenv import load_dotenv
from src.evaluation.evaluate import evaluate
from src.vector_db.vectordb import create_collection
from src.vector_db.ingest import ingest
from src.generation.pipeline import rag_pipeline
from src.core.config import EMBEDDING_PROVIDER

# 환경 변수 로드
load_dotenv()

if __name__ == "__main__":
    print("main 시작")

    create_collection(provider=EMBEDDING_PROVIDER)

    ingest()

    query = "AWS 리소스가 필요한 사업 찾아줘"

    result = rag_pipeline(query)

    print("\n===== 답변 =====\n")

    print(result)

    evaluate(
        evaluation_data=result, 
        model_name="gemini-3.1-flash-lite", 
        is_local=False
    )