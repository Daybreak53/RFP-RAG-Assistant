import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from src.parsing.run_parsing import run_parsing
from src.vector_db.ingest import ingest
from src.vector_db.vectordb import create_collection
from src.generation.chat import run_chat_mode
from src.generation.chat import run_single_query


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    load_dotenv()

    embed_provider  = cfg.providers.embedding
    llm_provider    = cfg.providers.llm
    collection_name = cfg.collection_name[embed_provider]

    print(f"[설정] 임베딩: {embed_provider} | LLM: {llm_provider}\n")

    run_all    = cfg.run_all
    run_parse  = run_all or cfg.run_parse
    run_ingest = run_all or cfg.run_ingest
    run_query  = run_all or cfg.run_query

    # [1] 문서 파싱
    rag_data = None
    if run_parse:

        print("--- [1] 문서 파싱 시작 ---")
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

    # [2] 벡터 DB 적재
    if run_ingest:
        if not rag_data:
            print("[경고] 파싱된 데이터가 없어 DB 적재를 수행할 수 없습니다.")
        else:
            print(f"--- [2] 벡터 DB 생성 및 데이터 삽입 시작 ({embed_provider}) ---")
            create_collection(
                embed_provider  = embed_provider,
                collection_name = collection_name,
            )
            ingest(
                embed_provider  = embed_provider,
                collection_name = collection_name,
                rag_data        = rag_data,
            )

    # [3-A] 대화형 모드
    if cfg.run_chat:
        print("--- [3] 대화 모드 시작 ---")
        run_chat_mode(cfg)
        return

    # [3-B] 단일 질의 모드
    if run_query:
        query_text = cfg.query
        print("--- [3] 단일 질의 시작 ---")
        print(f"질의: {query_text}\n")
        run_single_query(cfg=cfg, query_text=query_text)


if __name__ == "__main__":
    main()