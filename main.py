import argparse
import yaml
from dotenv import load_dotenv
from src.evaluation.evaluate import evaluate
from src.vector_db.vectordb import create_collection
from src.vector_db.ingest import ingest
from src.generation.pipeline import rag_pipeline
from src.parsing.run_parsing import run_parsing
from src.generation.pipeline import find_reference_for_query, rag_pipeline

def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    load_dotenv()
    
    # argparse 설정
    parser = argparse.ArgumentParser(description="RAG Pipeline Runner")
    parser.add_argument("--config", type=str, default="config.yaml", help="설정 파일 경로")
    
    # 실행 단계 제어 인자
    parser.add_argument("--all", action="store_true", help="전체 파이프라인 실행")
    parser.add_argument("--parse", action="store_true", help="문서 파싱 실행")
    parser.add_argument("--ingest", action="store_true", help="DB 세팅 및 적재 실행")
    parser.add_argument("--query", type=str, nargs="?", const="__use_yaml__", help="실행할 질의어 (입력 시 질의 단계 자동 실행)")
    parser.add_argument("--eval", action="store_true", help="평가 실행")
    
    # 변인 제어 인자
    parser.add_argument("--chunk_mode", type=str, help="청킹 방식 덮어쓰기 (recursive/semantic/sentence)")
    parser.add_argument("--chunk_size", type=int, help="청크 크기 덮어쓰기")
    parser.add_argument("--chunk_overlap", type=int, help="청크 오버랩 덮어쓰기")
    parser.add_argument("--match_threshold", type=float, help="match threshold 덮어쓰기")
    parser.add_argument("--embed_provider", type=str, help="임베딩 모델 덮어쓰기")
    parser.add_argument("--llm_provider", type=str, help="LLM 모델 덮어쓰기")
    parser.add_argument("--top_k", type=int, help="검색 결과 수 덮어쓰기")
    parser.add_argument("--score_threshold", type=float, help="유사도 임계값 덮어쓰기")
    parser.add_argument("--search_mode", type=str, help="검색 방식 덮어쓰기 (vector/hybrid)")

    args = parser.parse_args()
    
    # YAML 설정 로드
    config = load_config(args.config)
    
    # 우선순위 결정
    run_all = args.all or config['pipeline'].get('run_all', False)
    run_parse = run_all or args.parse or config['pipeline'].get('run_parse', False)
    run_ingest = run_all or args.ingest or config['pipeline'].get('run_ingest', False)
    
    if args.query == "__use_yaml__" or args.query is None and config['pipeline'].get('run_query', False):
        query_text = config.get('query')
    elif args.query and args.query != "__use_yaml__":
        query_text = args.query
    else:
        query_text = config.get('query')
        
    run_query = run_all or (args.query is not None) or config['pipeline'].get('run_query', False)
    run_eval = run_all or args.eval or config['pipeline'].get('run_eval', False)

    # 변인 설정
    chunk_mode = args.chunk_mode or config['parsing'].get("chunk_mode", "recursive")
    chunk_size = args.chunk_size or config['parsing'].get("chunk_size", 500)
    chunk_overlap = args.chunk_overlap or config['parsing'].get("chunk_overlap", 50)
    match_threshold = args.chunk_overlap or config['parsing'].get("match_threshold", 0.55)
    embed_provider = args.embed_provider or config['providers'].get('embedding', 'openai')
    llm_provider = args.llm_provider or config['providers'].get('llm', 'openai')
    collection_name = config['collection_name'].get(embed_provider, 'openai')
    top_k = args.top_k or config['retrieval'].get("top_k", 3)
    score_threshold = args.score_threshold or config['retrieval'].get("score_threshold", 0.2)
    search_mode = args.search_mode or config['retrieval'].get("search_mode", "vector")
    
    print(f"[설정] 임베딩: {embed_provider} | LLM: {llm_provider}\n")

    # 단계별 실행 로직
    if run_parse or run_ingest:
        rag_data = None
        if run_parse:
            from src.parsing.run_parsing import run_parsing

            print("--- [1] 문서 파싱 시작 ---")
            rag_data = run_parsing(
                chunk_mode=chunk_mode,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                match_threshold=match_threshold
            )
        
        if run_ingest:
            if not rag_data:
                print("[경고] 파싱된 데이터가 없어 DB 적재를 수행할 수 없습니다.")
            else:
                from src.vector_db.ingest import ingest
                from src.vector_db.vectordb import create_collection

                print(f"--- [2] 벡터 DB 생성 및 데이터 삽입 시작 ({embed_provider}) ---")
                create_collection(
                    embed_provider=embed_provider, 
                    collection_name=collection_name
                )
                ingest(
                    embed_provider=embed_provider,
                    collection_name=collection_name,
                    rag_data=rag_data
                )
            
    if run_query and query_text:
        reference = None
        if run_eval:
            reference = find_reference_for_query(query_text)
            if not reference:
                raise SystemExit(
                    "[평가 오류] --eval은 data/eval_dataset_hwp.json 또는 "
                    "data/eval_dataset_pdf.json의 user_input과 매칭되는 질의에서만 사용할 수 있습니다.\n"
                    f"현재 질의: {query_text}"
                )

        print(f"--- [3] RAG 파이프라인 질의 시작 ---")
        print(f"질의: {query_text}")
        
        result = rag_pipeline(
            collection_name=collection_name,
            embed_provider=embed_provider,
            llm_provider=llm_provider,
            query=query_text,
            top_k=top_k,
            score_threshold=score_threshold,
            search_mode=search_mode,
            reference=reference
        )
        
        print("\n===== 답변 =====")
        print(result["response"])
        print("===============\n")
        
        if run_eval:
            from src.evaluation.evaluate import evaluate

            print("--- [4] 평가 시작 ---")
            eval_config = config.get('evaluation', {})
            evaluate(
                evaluation_data=[result],
                model_name=eval_config.get('model_name', 'gpt-5-nano'),
                is_local=eval_config.get('is_local', False)
            )

if __name__ == "__main__":
    main()
