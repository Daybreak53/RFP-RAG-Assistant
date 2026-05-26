import logging
from typing import List, Dict, Any

from src.evaluation.evaluator import RagasEvaluator
from src.evaluation.logger import LangfuseEvalLogger

# 로거 설정
logger = logging.getLogger(__name__)


def _extract_contexts(retrieved_contexts: Any) -> List[str]:
    """
    다양한 형태(Dict, Object, String)의 검색된 문서 컨텍스트를 문자열 리스트로 정규화
    """
    if not retrieved_contexts:
        return []
        
    formatted_contexts = []
    for ctx in retrieved_contexts:
        if isinstance(ctx, dict):
            formatted_contexts.append(ctx.get("content", ""))
        elif hasattr(ctx, "content"):
            formatted_contexts.append(getattr(ctx, "content", ""))
        else:
            formatted_contexts.append(str(ctx))
            
    return formatted_contexts


def evaluate(
    evaluation_data: List[Dict[str, Any]], 
    model_name: str, 
    is_local: bool = False,
    langfuse: Any = None,
    generation: Any = None
) -> None:
    """
    RAG 파이프라인의 응답 결과를 Langfuse에 기록 및 Ragas를 통해 평가
    """
    if not langfuse or not generation:
        logger.warning("Langfuse 클라이언트나 Generation 객체가 제공되지 않아 평가를 건너뜁니다.")
        return

    trace_ids = []
    
    # Langfuse Generation 업데이트 (비용 및 토큰 사용량 등 기록)
    logger.info("Langfuse에 응답 결과 및 메타데이터를 업데이트합니다.")
    for data in evaluation_data:
        update_kwargs = {
            "input": data.get("user_input"),
            "output": data.get("response"),
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

    # RAGAS 평가용 데이터 포맷팅
    logger.info("RAGAS 평가를 위한 데이터 포맷팅을 진행합니다.")
    data_samples = {
        "user_input": [],
        "response": [],
        "contexts": [],
        "reference": [],
        "trace_id": trace_ids
    }
    
    for data in evaluation_data:
        data_samples["user_input"].append(data.get("user_input", ""))
        data_samples["response"].append(data.get("response", ""))
        
        raw_contexts = data.get("retrieved_contexts") or data.get("retrieved_context", [])
        data_samples["contexts"].append(_extract_contexts(raw_contexts))
        
        data_samples["reference"].append(data.get("reference", ""))

    # RAGAS 평가 실행
    logger.info(f"RagasEvaluator 초기화 및 평가 시작 (평가 모델: {model_name})")
    try:
        evaluator = RagasEvaluator(model_name)
        eval_output = evaluator.run_evaluation(data_samples)
        
        # 평가 결과 데이터프레임에 trace_id 매핑
        eval_output["trace_id"] = trace_ids
        
    except Exception as e:
        logger.error(f"RAGAS 평가 실행 중 오류가 발생했습니다: {e}", exc_info=True)
        return

    # Langfuse에 평가 결과 기록
    logger.info("평가된 지표(Metrics)를 Langfuse에 기록합니다.")
    try:
        eval_logger = LangfuseEvalLogger(langfuse=langfuse)
        eval_logger.log_evaluation_results(eval_results=eval_output)
    except Exception as e:
        logger.error(f"Langfuse 평가 결과 기록 중 오류가 발생했습니다: {e}", exc_info=True)