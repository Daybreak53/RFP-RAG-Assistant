from langfuse import get_client
from src.evaluation.evaluator import RagasEvaluator
from src.evaluation.logger import LangfuseEvalLogger

def evaluate():
    langfuse = get_client()
    
    trace_ids = []
    
    # Langfuse에 테스트용 Trace 생성
    with langfuse.start_as_current_observation(name="qa_test_1", as_type="span") as span:
        span.update(input="프랑스의 수도는?", output="파리입니다.")
        trace_ids.append(span.trace_id)
        
    with langfuse.start_as_current_observation(name="qa_test_2", as_type="span") as span:
        span.update(input="가장 큰 행성은?", output="목성입니다.")
        trace_ids.append(span.trace_id)

    langfuse.flush()

    # Ragas 평가용 데이터
    data_samples = {
        "user_input": ["프랑스의 수도는?", "가장 큰 행성은?"],
        "response": ["파리입니다.", "목성입니다."],
        "retrieved_contexts": [
            ["파리는 프랑스의 경제, 문화 중심지이다."],
            ["목성은 태양계 5번째 행성으로 가스 거성이다."]
        ],
        "reference": ["파리", "목성"],
        "trace_id": trace_ids
    }

    TARGET_PROVIDER = "openai" 
    
    # RAGAS 평가 실행
    evaluator = RagasEvaluator(provider=TARGET_PROVIDER)
    eval_results = evaluator.run_evaluation(data_samples)

    eval_results["trace_id"] = trace_ids

    # Langfuse에 평가 결과 기록
    logger = LangfuseEvalLogger()
    logger.log_evaluation_results(eval_results)