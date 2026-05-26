import argparse
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.retrieval.retriever import retrieve

# 환경 변수 로드
load_dotenv()

# 로거 설정
logger = logging.getLogger(__name__)

# 프로젝트 루트 기준으로 conf/config.yaml 기본 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "config.yaml"


def load_config(config_path: Path) -> dict:
    """
    YAML 설정 파일 로드
    """
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path.resolve()}")
        
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compare_search_modes(
    query: str, 
    config_path: Path = DEFAULT_CONFIG_PATH
) -> None:
    """
    여러 검색 모드(vector, keyword, hybrid)의 검색 결과를 비교 출력
    """
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {e}")
        return

    # 설정값 추출
    providers_cfg = config.get("providers", {})
    embed_provider = providers_cfg.get("embedding", "local")
    
    vector_db_cfg = config.get("vector_db", {})
    collection_name = vector_db_cfg.get("collection_names", {}).get(embed_provider)
    
    retrieval_cfg = config.get("retrieval", {})
    top_k = retrieval_cfg.get("top_k", 3)

    if not collection_name:
        logger.error(f"'{embed_provider}'에 대한 컬렉션 이름이 설정되어 있지 않습니다.")
        return

    modes = ["vector", "keyword", "hybrid"]

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  검색 모드: [{mode.upper()}]  |  쿼리: {query}")
        print(f"{'='*60}")

        try:
            results = retrieve(
                collection_name=collection_name,
                embed_provider=embed_provider,
                query=query,
                top_k=top_k,
                search_mode=mode,
            )
        except Exception as e:
            print(f"  [오류] 검색 중 문제가 발생했습니다: {e}")
            continue

        if not results:
            print("  검색 결과 없음")
            continue

        for rank, doc in enumerate(results, 1):
            score = doc.get('score', 0)
            file_name = doc.get('file_name', 'N/A')
            title = doc.get('title', 'N/A')
            section = doc.get('section_title', 'N/A')
            content_snippet = doc.get('content', '')[:80].strip()

            print(f"\n  [{rank}위] score={score:.4f}")
            print(f"       파일  : {file_name}")
            print(f"       제목  : {title}")
            print(f"       섹션  : {section}")
            print(f"       내용  : {content_snippet}...")


def main() -> None:
    """
    CLI 진입점
    """
    parser = argparse.ArgumentParser(description="RAG 검색 모드별(Vector/Keyword/Hybrid) 결과 비교 스크립트")
    parser.add_argument(
        "query", 
        nargs="?", 
        help="검색할 질의어 (생략 시 config.yaml의 'query' 값 사용)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=str(DEFAULT_CONFIG_PATH),
        help="config.yaml 파일의 명시적 경로"
    )
    
    args = parser.parse_args()
    config_path = Path(args.config)
    query = args.query

    # 쿼리가 CLI 인자로 주어지지 않은 경우 config.yaml에서 추출
    if not query:
        try:
            config = load_config(config_path)
            query = config.get("query", "").strip()
        except Exception as e:
            print(f"설정 파일에서 쿼리를 가져오는 중 오류 발생: {e}")
            return
            
    # 쿼리가 최종적으로 없는 경우 종료
    if not query:
        print("[알림] 검색할 쿼리가 제공되지 않았습니다. 인자나 config.yaml을 확인해주세요.")
        return
        
    # 검색 비교 실행
    compare_search_modes(query, config_path)


if __name__ == "__main__":
    main()