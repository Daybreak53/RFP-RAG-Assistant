from typing import List, Dict, Any
from src.evaluation.evaluator import RagasEvaluator
from src.evaluation.logger import LangfuseEvalLogger

def evaluate(
    evaluation_data: List[Dict[str, Any]], 
    model_name: str, 
    is_local: bool = False,
    langfuse=None,
    generation=None
):
    langfuse = langfuse

    trace_ids = []
    
    for idx, data in enumerate(evaluation_data):
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
        "contexts": [
            [
                ctx.get("content", "") if isinstance(ctx, dict)
                
                else getattr(ctx, "content", "") if hasattr(ctx, "content")
                
                else str(ctx)
                
                for ctx in d.get("retrieved_contexts", d.get("retrieved_context", []))
            ] 
            for d in evaluation_data
        ],
        "reference": [d["reference"] for d in evaluation_data],
        "trace_id": trace_ids
    }

    # RAGAS 평가 실행
    evaluator = RagasEvaluator(model_name)
    eval_output = evaluator.run_evaluation(data_samples)

    eval_output["trace_id"] = trace_ids

    # Langfuse에 평가 결과 기록
    logger = LangfuseEvalLogger(langfuse=langfuse)
    logger.log_evaluation_results(
        eval_results=eval_output
    )