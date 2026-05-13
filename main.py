from dotenv import load_dotenv
from src.evaluation.evaluate import evaluate

# 환경 변수 로드
load_dotenv()

if __name__ == "__main__":
    evaluate()