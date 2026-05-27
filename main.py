import logging
import sys

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.parsing.run_parsing import run_parsing
from src.vector_db.ingest import ingest
from src.vector_db.vectordb import create_collection
from src.generation.chat import run_chat_mode, run_single_query, run_multi_query
from src.evaluation.retrieval_metrics import run_retrieval_eval

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # 환경 변수 로드 및 체크
    if not load_dotenv():
        logger.warning(".env 파일을 찾을 수 없거나 로드할 환경 변수가 없습니다.")

    try:
        # 설정값 추출
        embed_provider  = cfg.providers.embedding
        llm_provider    = cfg.providers.llm
        collection_name = cfg.vector_db.collection_names[embed_provider]

        logger.info(f"파이프라인 설정 | 임베딩: {embed_provider} | LLM: {llm_provider}")

        # 실행 플래그 설정
        run_all             = cfg.pipeline.run_all
        run_parse           = run_all or cfg.pipeline.run_parse
        run_ingest          = run_all or cfg.pipeline.run_ingest
        run_query           = run_all or cfg.pipeline.run_query
        run_chat            = cfg.pipeline.run_chat
        run_retrieval_eval_ = cfg.pipeline.get("run_retrieval_eval", False)

        rag_data = None

        # [1] 문서 파싱
        if run_parse:
            logger.info("--- [1] 문서 파싱 시작 ---")
            p = cfg.parsing
            rag_data = run_parsing(
                chunk_mode          = p.chunk_mode,
                use_contextual      = p.use_contextual,
                chunk_size          = p.chunk_size,
                chunk_overlap       = p.chunk_overlap,
                semantic_threshold  = p.semantic_threshold,
                sem_rec_chunksize   = p.sem_rec_chunksize,
                sem_rec_overlap     = p.sem_rec_overlap,
                sentences_per_chunk = p.sentences_per_chunk,
                sentence_overlap    = p.sentence_overlap,
                match_threshold     = p.match_threshold,
                embed_provider      = embed_provider,
            )
            logger.info(f"문서 파싱 완료 (총 {len(rag_data) if rag_data else 0}개 청크)")

        # [2] 벡터 DB 적재
        if run_ingest:
            logger.info(f"--- [2] 벡터 DB 생성 및 데이터 삽입 시작 ({embed_provider}) ---")
            if not rag_data:
                logger.warning("파싱된 데이터가 없어 DB 적재를 수행할 수 없습니다. (run_parse를 활성화하거나 캐시를 확인하세요)")
            else:
                create_collection(
                    embed_provider  = embed_provider,
                    collection_name = collection_name,
                )
                ingest(
                    embed_provider  = embed_provider,
                    collection_name = collection_name,
                    rag_data        = rag_data,
                )
                logger.info("벡터 DB 적재 완료")

        # [3] 질의 실행
        if run_retrieval_eval_:
            logger.info("--- [3] Retrieval 일괄 평가 시작 ---")
            run_retrieval_eval(cfg)
        elif run_chat:
            logger.info("--- [3] 대화 모드 시작 ---")
            run_chat_mode(cfg)
        elif run_query:
            query_text = cfg.query
            logger.info(f"질의 내용: {query_text}")

            if cfg.retrieval.multi_query.enabled:
                logger.info("--- [3] Multi-Query 질의 시작 ---")
                run_multi_query(cfg=cfg, query_text=query_text)
            else:
                logger.info("--- [3] 단일 질의 시작 ---")
                run_single_query(cfg=cfg, query_text=query_text)
        else:
            logger.info("질의(run_query) / 대화(run_chat) / 검색평가(run_retrieval_eval) 모드가 모두 비활성화되어 파이프라인을 종료합니다.")

    except Exception as e:
        logger.error("파이프라인 실행 중 오류가 발생했습니다.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()