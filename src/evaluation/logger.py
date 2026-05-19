import pandas as pd

class LangfuseEvalLogger:
    def __init__(self, langfuse):
        self.langfuse = langfuse

    def log_evaluation_results(self, eval_results: pd.DataFrame, trace_id_col: str = "trace_id"):
        print("Langfuse에 Ragas 평가 지표를 기록하는 중...")
        
        input_col = 'user_input' if 'user_input' in eval_results.columns else 'question'
        ignore_cols = {input_col, 'response', 'retrieved_contexts', 'reference', 'answer', 'contexts', 'ground_truth', trace_id_col}

        for index, row in eval_results.iterrows():
            trace_id = row.get(trace_id_col)
            if pd.isna(trace_id):
                print(f"[경고] Trace ID가 누락되어 점수 기록을 건너뜁니다. (Index: {index})")
                continue
                
            for metric_name in eval_results.columns:
                if metric_name not in ignore_cols and pd.notna(row[metric_name]):
                    try:
                        self.langfuse.create_score(
                            trace_id=str(trace_id),
                            name=metric_name,
                            value=float(row[metric_name]),
                            data_type="NUMERIC",
                            comment=f"Input: {str(row.get(input_col, ''))[:30]}..."
                        )
                    except Exception as e:
                        print(f"[오류] {metric_name} 기록 중 문제 발생: {e}")

        self.langfuse.flush()
        print("Langfuse 기록이 완료되었습니다.")