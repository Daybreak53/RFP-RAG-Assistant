from typing import List, Dict, Any
from src.evaluation.evaluator import RagasEvaluator

def evaluate(
    evaluation_data: List[Dict[str, Any]], 
    model_name: str, 
    is_local: bool = False
):
    # Ragas 평가용 데이터 포맷팅
    data_samples = {
        "user_input": [d["user_input"] for d in evaluation_data],
        "response": [d["response"] for d in evaluation_data],
        "retrieved_contexts": [d["retrieved_context"] for d in evaluation_data],
        "reference": [d["reference"] for d in evaluation_data],
    }
    
    # RAGAS 평가 실행
    evaluator = RagasEvaluator(model_name)
    eval_output = evaluator.run_evaluation(data_samples)

    return eval_output