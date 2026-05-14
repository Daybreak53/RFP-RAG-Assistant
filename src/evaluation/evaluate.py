from typing import List, Dict, Any
from langfuse import get_client
from src.evaluation.evaluator import RagasEvaluator
from src.evaluation.logger import LangfuseEvalLogger

def evaluate(
    evaluation_data: List[Dict[str, Any]], 
    model_name: str, 
    is_local: bool = False
):
    langfuse = get_client()
    trace_ids = []
    
    for idx, data in enumerate(evaluation_data):
        with langfuse.start_as_current_observation(
            name=f"qa_evaluation_{idx}",
            as_type="generation",
            model=model_name
        ) as generation:
            update_kwargs = {
                "input": data["user_input"],
                "output": data["response"],
                "usage_details": data.get("usage", {})
            }
            
            if is_local:
                update_kwargs["cost_details"] = {
                    "input": 0.0,
                    "output": 0.0,
                    "total": 0.0
                }

            generation.update(**update_kwargs)
            trace_ids.append(generation.trace_id)

    langfuse.flush()

    # Ragas 평가용 데이터 포맷팅
    data_samples = {
        "user_input": [d["user_input"] for d in evaluation_data],
        "response": [d["response"] for d in evaluation_data],
        "retrieved_contexts": [[d["retrieved_context"]] for d in evaluation_data],
        "reference": [d["reference"] for d in evaluation_data],
        "trace_id": trace_ids
    }
    
    # RAGAS 평가 실행
    evaluator = RagasEvaluator(model_name)
    eval_output = evaluator.run_evaluation(data_samples)

    eval_output["trace_id"] = trace_ids

    # Langfuse에 평가 결과 기록
    logger = LangfuseEvalLogger()
    logger.log_evaluation_results(
        eval_output=eval_output,
        model_name=model_name,
    )