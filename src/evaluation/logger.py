import logging
from typing import Any
import pandas as pd

# 로거 설정
logger = logging.getLogger(__name__)

# Langfuse 스코어 기록 시 제외할 컬럼 목록
IGNORE_COLUMNS = {
    'user_input', 'question', 'response', 'retrieved_contexts', 
    'reference', 'answer', 'contexts', 'ground_truth'
}

class LangfuseEvalLogger:
    """
    Ragas 평가 결과를 Langfuse 플랫폼에 로깅(스코어링)
    """
    
    def __init__(self, langfuse: Any):
        self.langfuse = langfuse

    def log_evaluation_results(
        self, 
        eval_results: pd.DataFrame, 
        trace_id_col: str = "trace_id"
    ) -> None:
        """
        평가 지표가 담긴 데이터프레임을 순회하며 Langfuse에 스코어 기록
        """
        logger.info("Langfuse에 Ragas 평가 지표 기록을 시작합니다...")
        
        # 입력 데이터 컬럼명
        input_col = 'user_input'
        
        # 무시할 컬럼 목록에 추적 ID 컬럼을 추가
        skip_cols = IGNORE_COLUMNS.union({trace_id_col})

        for index, row in eval_results.iterrows():
            trace_id = row.get(trace_id_col)
            
            # Trace ID 유효성 검사 (NaN, None, 빈 문자열 방어)
            if pd.isna(trace_id) or not trace_id:
                logger.warning(f"Trace ID가 누락되어 점수 기록을 건너뜁니다. (DataFrame Index: {index})")
                continue
                
            # 입력 질문의 앞부분을 잘라서 Langfuse 코멘트로 활용
            input_text = str(row.get(input_col, ''))
            comment_text = f"Input: {input_text[:30]}..." if input_text else "Input: N/A"

            for metric_name in eval_results.columns:
                if metric_name not in skip_cols and pd.notna(row[metric_name]):
                    try:
                        self.langfuse.create_score(
                            trace_id=str(trace_id),
                            name=metric_name,
                            value=float(row[metric_name]),
                            data_type="NUMERIC",
                            comment=comment_text
                        )
                    except Exception as e:
                        logger.error(
                            f"지표 '{metric_name}' 기록 중 문제 발생 (Index: {index}): {e}", 
                            exc_info=True
                        )

        # 모든 스코어 기록 후 일괄 전송
        try:
            self.langfuse.flush()
            logger.info("Langfuse에 모든 평가 지표 기록이 완료되었습니다.")
        except Exception as e:
            logger.error(f"Langfuse flush 중 통신 오류 발생: {e}", exc_info=True)