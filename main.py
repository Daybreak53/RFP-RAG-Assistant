import argparse
import yaml
from dotenv import load_dotenv
from src.evaluation.evaluate import evaluate
from src.vector_db.vectordb import create_collection
from src.vector_db.ingest import ingest
from src.generation.pipeline import rag_pipeline
from src.parsing.run_parsing import main as run_parsing

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
    parser.add_argument("--query", type=str, help="실행할 질의어 (입력 시 질의 단계 자동 실행)")
    parser.add_argument("--eval", action="store_true", help="평가 실행")
    
    # 변인 제어 인자
    parser.add_argument("--embed_provider", type=str, help="임베딩 모델 덮어쓰기")
    parser.add_argument("--llm_provider", type=str, help="LLM 모델 덮어쓰기")

    args = parser.parse_args()
    
    # YAML 설정 로드
    config = load_config(args.config)
    
    # 우선순위 결정
    run_all = args.all or config['pipeline'].get('run_all', False)
    run_parse = run_all or args.parse or config['pipeline'].get('run_parse', False)
    run_ingest = run_all or args.ingest or config['pipeline'].get('run_ingest', False)
    
    query_text = args.query if args.query else config.get('query')
    run_query = run_all or bool(args.query) or config['pipeline'].get('run_query', False)
    run_eval = run_all or args.eval or config['pipeline'].get('run_eval', False)

    # 변인 설정
    embed_provider = args.embed_provider or config['providers'].get('embedding', 'openai')
    llm_provider = args.llm_provider or config['providers'].get('llm', 'openai')
    collection_name = config['collection_name'].get(embed_provider, 'openai')
    
    print(f"[설정] 임베딩: {embed_provider} | LLM: {llm_provider}\n")

    # 단계별 실행 로직
    if run_parse:
        print("--- [1] 문서 파싱 시작 ---")
        run_parsing()
        
    if run_ingest:
        print(f"--- [2] 벡터 DB 생성 및 데이터 삽입 시작 ({embed_provider}) ---")
        create_collection(
            embed_provider=embed_provider, 
            collection_name=collection_name
        )
        ingest(
            embed_provider=embed_provider,
            collection_name=collection_name
        ) 
        
    if run_query and query_text:
        print(f"--- [3] RAG 파이프라인 질의 시작 ---")
        print(f"질의: {query_text}")
        
        result = rag_pipeline(
            collection_name=collection_name,
            embed_provider=embed_provider,
            llm_provider=llm_provider,
            query=query_text
        )
        
        print("\n===== 답변 =====")
        print(result["response"])
        print("===============\n")
        
        if run_eval:
            print("--- [4] 평가 시작 ---")
            eval_config = config.get('evaluation', {})
            evaluate(
                evaluation_data=[result],
                model_name=eval_config.get('model_name', 'gpt-5-nano'),
                is_local=eval_config.get('is_local', False)
            )

if __name__ == "__main__":
    main()